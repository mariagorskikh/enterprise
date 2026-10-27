import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from posixpath import dirname
from typing import Any, AsyncGenerator, ClassVar

import base62
import httpx
from fastapi import Request
from pydantic import Field, SecretStr

from openhands.app_server.errors import SandboxDeleteRetryError, SandboxError
from openhands.app_server.sandbox.managed_sandbox_service import (
    ManagedSandboxService,
    ManagedSandboxServiceInjector,
    ProviderOutcome,
)
from openhands.app_server.sandbox.maritime_sandbox_spec_service import (
    MaritimeSandboxSpecInfo,
)
from openhands.app_server.sandbox.sandbox_models import (
    AGENT_SERVER,
    ExposedUrl,
    SandboxInfo,
    SandboxPage,
    SandboxRecord,
    SandboxStatus,
)
from openhands.app_server.sandbox.sandbox_service import SandboxService
from openhands.app_server.sandbox.sandbox_spec_models import SandboxSpecInfo
from openhands.app_server.sandbox.sandbox_spec_service import (
    SandboxSpecService,
    get_agent_server_env,
    resolve_sandbox_spec,
)
from openhands.app_server.sandbox.sandbox_store import (
    MARITIME_BACKEND,
    StoredSandbox,
    get_stored_sandbox_by_session_api_key,
    hash_session_api_key,
    require_user_id,
    search_stored_sandboxes,
)
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.specifiy_user_context import ADMIN

_logger = logging.getLogger(__name__)

# Ownership lives in the sandbox table (see `sandbox_store`). The agent's name
# prefix and external id tag managed agents so that one with no row can be
# found, and keep a user's listing narrow.
AGENT_NAME_PREFIX = 'openhands-'
EXTERNAL_ID_PREFIX = 'openhands:'

# Maritime routes an agent's URL to its public port, which the micro-VM
# forwards to $PORT. The image binds $PORT, so this is the only port to declare.
PUBLIC_PORT = 8080

ACTIVE = 'active'
SLEEPING = 'sleeping'
STOPPED = 'stopped'
DEPLOYING = 'deploying'
ERROR = 'error'

STATUS_MAPPING = {
    ACTIVE: SandboxStatus.RUNNING,
    SLEEPING: SandboxStatus.PAUSED,
    STOPPED: SandboxStatus.PAUSED,
    DEPLOYING: SandboxStatus.STARTING,
    ERROR: SandboxStatus.ERROR,
}

MARITIME_AUTH_FAILURE = (
    'Maritime rejected the API key. Check MARITIME_API_KEY: it must be a '
    'Maritime API key (mk_...) with the provision and deploy scopes.'
)
MARITIME_PLAN_LIMIT = (
    'The Maritime plan has no room for another awake agent. Pause or delete an '
    'idle sandbox, or raise the plan limits at maritime.sh.'
)


class MaritimeApiError(SandboxError):
    """A Maritime API call failed with an HTTP status."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _external_id(user_id: str) -> str:
    return f'{EXTERNAL_ID_PREFIX}{user_id}'


def _as_maritime_spec(sandbox_spec: SandboxSpecInfo) -> MaritimeSandboxSpecInfo:
    """Narrow a spec to the Maritime shape, filling defaults for a plain spec."""
    if isinstance(sandbox_spec, MaritimeSandboxSpecInfo):
        return sandbox_spec
    return MaritimeSandboxSpecInfo(**sandbox_spec.model_dump())


@dataclass
class MaritimeClient:
    """The calls the Maritime backend makes to the Maritime API.

    Errors come back as ``MaritimeApiError`` carrying the HTTP status. A
    rejected key is reported as itself, so it is never mistaken for a missing
    agent.
    """

    httpx_client: httpx.AsyncClient
    api_key: str
    api_url: str

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> Any:
        try:
            response = await self.httpx_client.request(
                method,
                f'{self.api_url.rstrip("/")}{path}',
                json=json,
                params=params,
                headers={'Authorization': f'Bearer {self.api_key}'},
                timeout=60,
            )
        except httpx.HTTPError as exc:
            raise MaritimeApiError(f'Could not reach Maritime: {exc}') from exc
        if response.status_code in (401, 403):
            raise MaritimeApiError(
                f'{MARITIME_AUTH_FAILURE} ({response.status_code}: {_detail(response)})',
                response.status_code,
            )
        if response.status_code >= 400:
            raise MaritimeApiError(
                f'Maritime answered {response.status_code} to {method} {path}: '
                f'{_detail(response)}',
                response.status_code,
            )
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    async def create_agent(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request('POST', '/api/agents', json=body)

    async def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        """Maritime's record of an agent, or None when Maritime has none."""
        try:
            return await self._request('GET', f'/api/agents/{agent_id}')
        except MaritimeApiError as exc:
            if exc.status_code == 404:
                return None
            raise

    async def list_agents(self, external_id: str | None = None) -> list[dict[str, Any]]:
        params = {'externalId': external_id} if external_id else None
        agents = await self._request('GET', '/api/agents', params=params)
        return [agent for agent in agents or [] if isinstance(agent, dict)]

    async def start_agent(self, agent_id: str) -> dict[str, Any] | None:
        """Wake a sleeping agent, or start a stopped one."""
        return await self._request('POST', f'/api/agents/{agent_id}/start')

    async def sleep_agent(self, agent_id: str) -> None:
        await self._request('POST', f'/api/agents/{agent_id}/sleep')

    async def delete_agent(self, agent_id: str) -> None:
        """Delete an agent. One Maritime no longer has counts as deleted."""
        try:
            await self._request('DELETE', f'/api/agents/{agent_id}')
        except MaritimeApiError as exc:
            if exc.status_code != 404:
                raise


@dataclass
class MaritimeSandboxService(ManagedSandboxService):
    """Sandbox service backed by Maritime agents, one micro-VM per sandbox.

    Each sandbox is a Maritime agent running the agent server image in
    deferred-init mode, reached at the agent's own HTTPS and WebSocket URL
    (``https://{agent_id}.mtm.sh``). ``start_sandbox`` completes the
    ``/api/init`` handshake before returning.

    Ownership, spec identity and the session API key live in the sandbox
    table (see ``sandbox_store``). The session API key doubles as the init key
    and the agent server's secret key: Maritime keeps it as the agent's secret
    ``OH_SECRET_KEY``, so an agent server that boots dormant again after a
    cold restart can be claimed with the key already on the row, and the
    secrets it persisted still decrypt.

    Maritime pauses an agent by snapshotting its micro-VM, memory and
    processes included, and wakes it on the next request to its URL. Like E2B,
    it needs a publicly reachable ``OH_WEB_URL`` for agent server event
    callbacks.
    """

    backend: ClassVar[str] = MARITIME_BACKEND

    sandbox_spec_service: SandboxSpecService
    maritime: MaritimeClient
    agent_url_domain: str
    idle_ttl_seconds: int
    init_timeout_seconds: int
    poll_interval: float
    web_url: str | None = None
    permitted_cors_origins: list[str] = field(default_factory=list)

    async def _live_agents(self, wanted_ids: set[str]) -> dict[str, dict[str, Any]]:
        """Maritime's records for the sandboxes on a page, indexed by id.

        The rows have already decided ownership. The external id filter only
        narrows the list to the caller's agents. ``ADMIN`` reads across users,
        so it lists every agent the key can see.
        """
        if not wanted_ids:
            return {}
        external_id = None
        if self.user_context != ADMIN:
            external_id = _external_id(await require_user_id(self.user_context))
        agents = await self.maritime.list_agents(external_id)
        return {agent['id']: agent for agent in agents if agent.get('id') in wanted_ids}

    # ------------------------------------------------------------------
    # Info mapping
    # ------------------------------------------------------------------

    async def _get_spec(self, sandbox_spec_id: str) -> MaritimeSandboxSpecInfo:
        """Get the spec a sandbox was created from, or the default shape.

        A sandbox outlives an edit to the configured spec list, so a spec that
        is no longer offered still needs a working dir to build its init body.
        """
        sandbox_spec = await self.sandbox_spec_service.get_sandbox_spec(sandbox_spec_id)
        if sandbox_spec is None:
            return MaritimeSandboxSpecInfo(id=sandbox_spec_id, command=None)
        return _as_maritime_spec(sandbox_spec)

    def _agent_url(self, agent_id: str, agent: dict[str, Any] | None = None) -> str:
        """The agent's own URL, without a trailing slash.

        Taken from Maritime's ``webUrl`` when the record carries one, and built
        from the agent URL domain otherwise.
        """
        web_url = agent.get('webUrl') if agent else None
        if not web_url:
            web_url = f'https://{agent_id}.{self.agent_url_domain}'
        return web_url.rstrip('/')

    def _exposed_urls(
        self, agent_id: str, agent: dict[str, Any] | None = None
    ) -> list[ExposedUrl]:
        # Maritime publishes one port per agent, so VSCode and the worker
        # ports are not reachable from outside and are not offered.
        return [
            ExposedUrl(
                name=AGENT_SERVER,
                url=self._agent_url(agent_id, agent),
                port=PUBLIC_PORT,
            )
        ]

    def _to_sandbox_info(
        self,
        stored_sandbox: StoredSandbox,
        agent: dict[str, Any] | None,
        session_api_key: str | None = None,
    ) -> SandboxInfo:
        """Build a SandboxInfo from the stored row plus Maritime's record.

        A row Maritime no longer knows about is MISSING, which drives the
        archived-conversation UI.
        """
        status = (
            SandboxStatus.MISSING
            if agent is None
            else STATUS_MAPPING.get(str(agent.get('status')), SandboxStatus.ERROR)
        )
        exposed_urls = None
        if status == SandboxStatus.RUNNING and session_api_key:
            exposed_urls = self._exposed_urls(stored_sandbox.id, agent)
        else:
            session_api_key = None

        return SandboxInfo(
            id=stored_sandbox.id,
            created_by_user_id=stored_sandbox.created_by_user_id,
            sandbox_spec_id=stored_sandbox.sandbox_spec_id,
            status=status,
            session_api_key=session_api_key,
            exposed_urls=exposed_urls,
            created_at=stored_sandbox.created_at,
        )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    @staticmethod
    def _raw_key(stored_sandbox: StoredSandbox) -> str | None:
        """The session API key on a row, decrypted."""
        key = stored_sandbox.session_api_key
        return key.get_secret_value() if key else None

    async def search_sandboxes(
        self,
        page_id: str | None = None,
        limit: int = 100,
    ) -> SandboxPage:
        """Search for sandboxes.

        One query against the sandbox table for the page, then one Maritime
        list call for the live state of everything on it.
        """
        page = await search_stored_sandboxes(
            self.db_session, self.user_context, MARITIME_BACKEND, page_id, limit
        )
        agents = await self._live_agents({row.id for row in page.items})
        return SandboxPage(
            items=[
                self._to_sandbox_info(
                    stored_sandbox,
                    agents.get(stored_sandbox.id),
                    self._raw_key(stored_sandbox),
                )
                for stored_sandbox in page.items
            ],
            next_page_id=page.next_page_id,
        )

    async def get_sandbox(self, sandbox_id: str) -> SandboxInfo | None:
        """Get a single sandbox."""
        stored_sandbox = await self._get_stored_sandbox(sandbox_id)
        if stored_sandbox is None:
            return None
        return self._to_sandbox_info(
            stored_sandbox,
            await self.maritime.get_agent(sandbox_id),
            self._raw_key(stored_sandbox),
        )

    async def get_sandbox_by_session_api_key(
        self, session_api_key: str
    ) -> SandboxInfo | None:
        """Get a single sandbox by session API key, on the hash index."""
        stored_sandbox = await get_stored_sandbox_by_session_api_key(
            self.db_session, self.user_context, MARITIME_BACKEND, session_api_key
        )
        if stored_sandbox is None:
            return None
        return self._to_sandbox_info(
            stored_sandbox,
            await self.maritime.get_agent(stored_sandbox.id),
            self._raw_key(stored_sandbox),
        )

    async def get_sandbox_record_by_session_api_key(
        self, session_api_key: str
    ) -> SandboxRecord | None:
        """Get sandbox identity by session API key.

        An indexed lookup with no Maritime call. This runs on the webhook path,
        once per batch of agent events.
        """
        stored_sandbox = await get_stored_sandbox_by_session_api_key(
            self.db_session, self.user_context, MARITIME_BACKEND, session_api_key
        )
        if stored_sandbox is None:
            return None
        return SandboxRecord(
            id=stored_sandbox.id,
            created_by_user_id=stored_sandbox.created_by_user_id,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _create_request(
        self,
        sandbox_spec: MaritimeSandboxSpecInfo,
        user_id: str,
        session_api_key: str,
    ) -> dict[str, Any]:
        """Build the ``POST /api/agents`` body.

        ``publicWeb`` lets the browser and the app server reach the agent URL
        without a Maritime login. What protects the agent server is its own
        auth: the dormant server answers only a caller holding the init key,
        and once claimed only one holding the session API key.
        """
        body: dict[str, Any] = {
            'name': f'{AGENT_NAME_PREFIX}{base62.encodebytes(os.urandom(6)).lower()}',
            'description': f'OpenHands sandbox ({sandbox_spec.id})',
            'externalId': _external_id(user_id),
            'framework': 'custom',
            'imageName': sandbox_spec.id,
            'exposedPort': PUBLIC_PORT,
            'publicWeb': True,
            'healthCheckPath': '/alive',
            'memMb': sandbox_spec.mem_mb,
            'vcpus': sandbox_spec.vcpus,
            'idleTtlSeconds': self.idle_ttl_seconds,
            'initialEnvVars': [
                {'key': 'OH_DEFERRED_INIT', 'value': '1', 'isSecret': False},
                {'key': 'OH_SECRET_KEY', 'value': session_api_key, 'isSecret': True},
            ],
        }
        if sandbox_spec.disk_gb is not None:
            body['diskGb'] = sandbox_spec.disk_gb
        return body

    async def start_sandbox(
        self,
        sandbox_spec_id: str | None = None,
        sandbox_id: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> SandboxInfo:
        """Start a new sandbox and initialize its agent server.

        ``sandbox_id`` is ignored: Maritime assigns the agent id and
        ``SandboxInfo.id`` is that id.

        The ``/api/init`` handshake completes before this returns. ``/alive``,
        ``/health`` and ``/ready`` answer 200 while the server is still
        dormant, so a caller waiting on readiness would otherwise be handed a
        server that rejects every ``/api`` call with a 503.
        """
        # Every sandbox has an owner. Check before pause_old_sandboxes, which
        # would reach every user's sandboxes as ADMIN.
        user_id = await require_user_id(self.user_context)

        # Enforce sandbox limits by pausing old sandboxes
        await self.pause_old_sandboxes(self.max_num_sandboxes - 1)

        user_default_spec_id = await self.user_context.get_default_sandbox_spec_id()
        sandbox_spec = _as_maritime_spec(
            await resolve_sandbox_spec(
                sandbox_spec_id,
                user_default_spec_id,
                self.sandbox_spec_service,
                _logger,
            )
        )

        session_api_key = base62.encodebytes(os.urandom(32))
        try:
            agent = await self.maritime.create_agent(
                self._create_request(sandbox_spec, user_id, session_api_key)
            )
        except MaritimeApiError as exc:
            _logger.exception('Failed to create Maritime agent', stack_info=True)
            if exc.status_code == 429:
                raise SandboxError(f'{MARITIME_PLAN_LIMIT} ({exc})') from exc
            raise SandboxError(f'Failed to start sandbox: {exc}') from exc

        agent_id = agent['id']
        stored_sandbox = self._new_stored_sandbox(
            id=agent_id,
            created_by_user_id=user_id,
            sandbox_spec_id=sandbox_spec.id,
            session_api_key_hash=hash_session_api_key(session_api_key),
            session_api_key=SecretStr(session_api_key),
        )
        try:
            # Maritime assigns the id, so the row can only be written after
            # create. An agent whose row fails to write is deleted with the rest.
            self.db_session.add(stored_sandbox)
            await self.db_session.flush()
            agent = await self._wait_for_active(agent_id)
            await self._initialize_agent_server(
                agent_id, agent, sandbox_spec, session_api_key, extra_env=extra_env
            )
        except BaseException:
            # A half-started agent is deleted rather than reasoned about: init
            # rotates the server's keys, so a retry could not tell a wrong key
            # from a used one.
            await self._delete_quietly(agent_id)
            raise

        return SandboxInfo(
            id=agent_id,
            created_by_user_id=user_id,
            sandbox_spec_id=sandbox_spec.id,
            status=SandboxStatus.RUNNING,
            session_api_key=session_api_key,
            exposed_urls=self._exposed_urls(agent_id, agent),
            created_at=stored_sandbox.created_at,
        )

    async def _delete_quietly(self, agent_id: str) -> None:
        """Delete an agent on a failure path, without masking the original error."""
        try:
            await self.maritime.delete_agent(agent_id)
        except Exception:
            _logger.warning(
                f'Could not delete Maritime agent {agent_id} after a failed start'
            )

    async def _wait_for_active(self, agent_id: str) -> dict[str, Any]:
        """Poll Maritime until the agent's micro-VM is up.

        The first agent on a host converts the image to a root filesystem,
        which takes longest; later ones start from the cache.
        """
        deadline = time.monotonic() + self.init_timeout_seconds
        while True:
            agent = await self.maritime.get_agent(agent_id)
            if agent is None:
                raise SandboxError(
                    f'Maritime agent {agent_id} disappeared while starting'
                )
            status = agent.get('status')
            if status == ACTIVE:
                return agent
            if status == ERROR:
                raise SandboxError(
                    f'Maritime agent {agent_id} failed to start. Its deploy log is '
                    'on the agent page at maritime.sh.'
                )
            if time.monotonic() >= deadline:
                raise SandboxError(
                    f'Maritime agent {agent_id} did not start within '
                    f'{self.init_timeout_seconds}s (status {status!r})'
                )
            await asyncio.sleep(self.poll_interval)

    async def _initialize_agent_server(
        self,
        agent_id: str,
        agent: dict[str, Any] | None,
        sandbox_spec: MaritimeSandboxSpecInfo,
        session_api_key: str,
        allow_initialized: bool = False,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        """Wait for the dormant agent server and hand it its runtime config.

        With ``allow_initialized``, an agent server that is already
        initialized is left alone.
        """
        agent_server_url = self._agent_url(agent_id, agent)
        if not await self._wait_for_dormant(agent_server_url, allow_initialized):
            return
        try:
            response = await self.httpx_client.post(
                f'{agent_server_url}/api/init',
                json=self._build_init_request(sandbox_spec, session_api_key, extra_env),
                headers={'X-Init-API-Key': session_api_key},
            )
        except httpx.HTTPError as exc:
            raise SandboxError(
                f'Could not reach the agent server for sandbox {agent_id}'
            ) from exc
        if response.status_code != 200:
            _logger.error(
                f'Agent server init for sandbox {agent_id} returned '
                f'{response.status_code}: {response.text}'
            )
            if response.status_code == 401:
                raise SandboxError(
                    f'The agent server in sandbox {agent_id} rejected the init '
                    "API key. It must match the agent's OH_SECRET_KEY on Maritime."
                )
            raise SandboxError(f'Failed to initialize sandbox {agent_id}')

    async def _wait_for_dormant(
        self, agent_server_url: str, allow_initialized: bool
    ) -> bool:
        """Poll ``GET /api/init`` until the agent server can be claimed.

        Returns True when it is dormant, and False when it is already
        initialized and ``allow_initialized`` is set. Maritime answers 503
        while the micro-VM boots or wakes, which is expected and not fatal.
        """
        deadline = time.monotonic() + self.init_timeout_seconds
        while True:
            try:
                response = await self.httpx_client.get(f'{agent_server_url}/api/init')
                if response.status_code == 200:
                    state = response.json().get('state')
                    if state == 'dormant':
                        return True
                    if state == 'ready':
                        if allow_initialized:
                            return False
                        raise SandboxError(
                            f'Agent server at {agent_server_url} is already initialized'
                        )
            except (httpx.HTTPError, ValueError) as exc:
                _logger.debug(f'Waiting for {agent_server_url}: {exc}')
            if time.monotonic() >= deadline:
                raise SandboxError(
                    f'Agent server at {agent_server_url} did not become ready '
                    f'within {self.init_timeout_seconds}s'
                )
            await asyncio.sleep(self.poll_interval)

    def _build_init_request(
        self,
        sandbox_spec: MaritimeSandboxSpecInfo,
        session_api_key: str,
        extra_env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Build the ``POST /api/init`` body.

        The agent server forbids extra fields, so this carries only what it
        accepts.
        """
        # The spec's working dir is the project checkout; its parent is the
        # workspace root the agent server keeps its own state under. None of
        # these directories need to exist beforehand.
        working_dir = sandbox_spec.working_dir.rstrip('/')
        workspace_dir = dirname(working_dir)
        if workspace_dir in ('', '/'):
            workspace_dir = working_dir
        body: dict[str, Any] = {
            'session_api_keys': [session_api_key],
            # It lives as long as the session key, so the secrets the agent
            # server persisted still decrypt when a cold restart has to repeat
            # this handshake.
            'secret_key': session_api_key,
            'conversations_path': f'{workspace_dir}/conversations',
            'bash_events_dir': f'{workspace_dir}/bash_events',
            'conversation_worktree_root': f'{workspace_dir}/worktrees',
            # `get_agent_server_env` is resolved here rather than baked into
            # the spec's initial_env, because it can carry LLM_API_KEY and the
            # spec is serialized verbatim by the public sandbox-specs endpoint.
            'env': {
                **sandbox_spec.initial_env,
                **get_agent_server_env(),
            },
        }
        body['env'].update(self._lifecycle_env())
        if extra_env:
            body['env'].update(extra_env)

        cors_origins = []
        if self.web_url:
            cors_origins.append(self.web_url)
        cors_origins.extend(self.permitted_cors_origins)
        if cors_origins:
            body['allow_cors_origins'] = list(dict.fromkeys(cors_origins))

        if self.web_url and 'localhost' not in self.web_url:
            body['webhooks'] = [{'base_url': f'{self.web_url}/api/v1/webhooks'}]
        else:
            _logger.warning(
                'No publicly reachable OH_WEB_URL is configured, so Maritime '
                'sandboxes cannot post events back to the app server. '
                'Conversations will start but will not receive agent events.'
            )

        return body

    async def _resume_at_provider(
        self, stored_sandbox: StoredSandbox
    ) -> ProviderOutcome:
        """Wake the agent, then claim its agent server again if it needs it.

        A sleeping agent wakes from its memory snapshot holding the key it was
        claimed with, so the handshake is skipped. An agent Maritime had to
        restart cold boots dormant, so the handshake is repeated with the key
        on the row.
        """
        sandbox_id = stored_sandbox.id
        agent = await self.maritime.get_agent(sandbox_id)
        if agent is None:
            return ProviderOutcome.FAILED
        status = agent.get('status')
        if status == DEPLOYING:
            return ProviderOutcome.SKIPPED
        if status == ERROR:
            return ProviderOutcome.FAILED
        session_api_key = self._raw_key(stored_sandbox)
        if session_api_key is None:
            return ProviderOutcome.FAILED

        changed = status != ACTIVE
        if changed:
            try:
                agent = await self.maritime.start_agent(sandbox_id)
            except MaritimeApiError as exc:
                if exc.status_code == 429:
                    raise SandboxError(f'{MARITIME_PLAN_LIMIT} ({exc})') from exc
                if exc.status_code == 404:
                    return ProviderOutcome.FAILED
                _logger.exception(
                    f'Error resuming sandbox {sandbox_id}', stack_info=True
                )
                return ProviderOutcome.FAILED
            if not agent or agent.get('status') != ACTIVE:
                agent = await self._wait_for_active(sandbox_id)

        await self._initialize_agent_server(
            sandbox_id,
            agent,
            await self._get_spec(stored_sandbox.sandbox_spec_id),
            session_api_key,
            allow_initialized=True,
        )
        return ProviderOutcome.CHANGED if changed else ProviderOutcome.ALREADY_DONE

    async def _pause_at_provider(
        self, stored_sandbox: StoredSandbox
    ) -> ProviderOutcome:
        """Put the agent to sleep: Maritime snapshots its micro-VM.

        The stored key is kept, because the agent wakes holding the same key
        (see ``_resume_at_provider``).
        """
        sandbox_id = stored_sandbox.id
        agent = await self.maritime.get_agent(sandbox_id)
        if agent is None:
            return ProviderOutcome.FAILED
        status = agent.get('status')
        if status in (SLEEPING, STOPPED):
            return ProviderOutcome.ALREADY_DONE
        if status != ACTIVE:
            return ProviderOutcome.SKIPPED
        try:
            await self.maritime.sleep_agent(sandbox_id)
        except MaritimeApiError:
            _logger.exception(f'Error pausing sandbox {sandbox_id}', stack_info=True)
            return ProviderOutcome.FAILED
        return ProviderOutcome.CHANGED

    async def _delete_at_provider(self, stored_sandbox: StoredSandbox) -> None:
        """Delete the agent, its micro-VM and its volume.

        An agent Maritime no longer has counts as deleted, so its row is
        removed rather than kept for a retry with nothing left to do.
        """
        sandbox_id = stored_sandbox.id
        try:
            await self.maritime.delete_agent(sandbox_id)
        except MaritimeApiError as exc:
            _logger.exception(f'Error deleting sandbox {sandbox_id}', stack_info=True)
            raise SandboxDeleteRetryError(
                f'Could not complete delete for sandbox {sandbox_id}: {exc}'
            ) from exc


def _detail(response: httpx.Response) -> str:
    """The error detail Maritime returns, or the raw body."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(body, dict) and 'detail' in body:
        return str(body['detail'])
    return str(body)[:500]


class MaritimeSandboxServiceInjector(ManagedSandboxServiceInjector):
    """Dependency injector for Maritime sandbox services."""

    backend: ClassVar[str] = MARITIME_BACKEND

    api_key: str = Field(
        default_factory=lambda: os.getenv('MARITIME_API_KEY', ''),
        description=(
            'The Maritime API key (mk_...), with the provision and deploy '
            'scopes. Defaults to the MARITIME_API_KEY env var.'
        ),
    )
    api_url: str = Field(
        default_factory=lambda: os.getenv(
            'MARITIME_API_URL', 'https://api.maritime.sh'
        ),
        description='The Maritime API URL. Defaults to the MARITIME_API_URL env var.',
    )
    agent_url_domain: str = Field(
        default_factory=lambda: os.getenv('MARITIME_AGENT_URL_DOMAIN', 'mtm.sh'),
        description=(
            'The domain agent URLs live under, as https://{agent_id}.{domain}. '
            'Used only when Maritime returns no webUrl.'
        ),
    )
    idle_ttl_seconds: int | None = Field(
        default_factory=lambda: (
            int(os.environ['MARITIME_IDLE_TTL_SECONDS'])
            if os.getenv('MARITIME_IDLE_TTL_SECONDS')
            else None
        ),
        description=(
            'Seconds without a request after which Maritime puts an agent to '
            "sleep on its own. Defaults to the lifecycle's max session (12 "
            "hours), so the lifecycle worker, which reads the agent server's "
            'own idle time, does the pausing and this is only a backstop.'
        ),
    )
    max_num_sandboxes: int = Field(
        default=10,
        description='Maximum number of sandboxes allowed to run simultaneously',
    )
    init_timeout_seconds: int = Field(
        default=600,
        description=(
            'The max time to wait for a new agent to start and its agent server '
            'to reach the dormant state. The first agent on a Maritime host '
            'converts the image to a root filesystem, which takes minutes.'
        ),
    )
    poll_interval: float = Field(
        default=2.0,
        description='Seconds between polls while waiting for an agent to start',
    )

    def _effective_idle_ttl(self) -> int:
        if self.idle_ttl_seconds is not None:
            return self.idle_ttl_seconds
        return self.lifecycle.max_session_seconds or 43200

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[SandboxService, None]:
        # Define inline to prevent circular lookup
        from openhands.app_server.config import (
            get_db_session,
            get_global_config,
            get_httpx_client,
            get_sandbox_spec_service,
            get_user_context,
        )

        config = get_global_config()
        async with (
            get_user_context(state, request) as user_context,
            get_httpx_client(state, request) as httpx_client,
            get_sandbox_spec_service(state, request) as sandbox_spec_service,
            get_db_session(state, request) as db_session,
        ):
            yield MaritimeSandboxService(
                sandbox_spec_service=sandbox_spec_service,
                user_context=user_context,
                httpx_client=httpx_client,
                db_session=db_session,
                maritime=MaritimeClient(
                    httpx_client=httpx_client,
                    api_key=self.api_key,
                    api_url=self.api_url,
                ),
                agent_url_domain=self.agent_url_domain,
                idle_ttl_seconds=self._effective_idle_ttl(),
                max_num_sandboxes=self.max_num_sandboxes,
                init_timeout_seconds=self.init_timeout_seconds,
                poll_interval=self.poll_interval,
                web_url=config.web_url,
                permitted_cors_origins=config.permitted_cors_origins,
                lifecycle=self.lifecycle,
            )
