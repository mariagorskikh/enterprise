"""Tests for MaritimeSandboxService.

Maritime is replaced by an in-memory fake of ``MaritimeClient``; the agent
server by a fake that records the requests the service makes to it. The HTTP
client itself is tested against ``httpx.MockTransport``. Focus areas:
- the create request: image, public port, size, owner tag and secret env
- the /api/init handshake start_sandbox completes before returning
- the sandbox table as the store for ownership, spec identity and the key
- status mapping, including MISSING, and the single exposed URL
- pause, resume (including a cold restart that boots dormant) and delete
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import select

from openhands.app_server.errors import (
    AuthError,
    SandboxDeleteRetryError,
    SandboxError,
)
from openhands.app_server.sandbox.lifecycle.settings import SandboxLifecycleSettings
from openhands.app_server.sandbox.maritime_sandbox_service import (
    EXTERNAL_ID_PREFIX,
    MARITIME_AUTH_FAILURE,
    PUBLIC_PORT,
    MaritimeApiError,
    MaritimeClient,
    MaritimeSandboxService,
    MaritimeSandboxServiceInjector,
)
from openhands.app_server.sandbox.maritime_sandbox_spec_service import (
    MaritimeSandboxSpecInfo,
)
from openhands.app_server.sandbox.preset_sandbox_spec_service import (
    PresetSandboxSpecService,
)
from openhands.app_server.sandbox.sandbox_models import AGENT_SERVER, SandboxStatus
from openhands.app_server.sandbox.sandbox_store import (
    MARITIME_BACKEND,
    StoredSandbox,
    hash_session_api_key,
)
from openhands.app_server.user.specifiy_user_context import ADMIN
from openhands.app_server.user.user_context import UserContext

IMAGE = 'ghcr.io/maritime-sh/openhands-agent-server:1.58.1'
OWNER_ID = 'user-1'
OTHER_USER_ID = 'user-2'
AGENT_ID = 'c25053e3-dcab-4d9c-abdc-0117c5decf67'
OTHER_AGENT_ID = '9f1d2e3c-0000-4000-8000-000000000002'
WEB_URL = 'https://app.example.com'
AGENT_URL = f'https://{AGENT_ID}.mtm.sh'
SESSION_API_KEY = 'the-session-key'
CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)
API_URL = 'https://api.maritime.example.com'


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _agent(
    agent_id: str = AGENT_ID,
    status: str = 'active',
    user_id: str = OWNER_ID,
) -> dict[str, Any]:
    """The subset of Maritime's AgentResponse the service reads."""
    return {
        'id': agent_id,
        'name': 'openhands-abc',
        'status': status,
        'externalId': f'{EXTERNAL_ID_PREFIX}{user_id}',
        'webUrl': f'https://{agent_id}.mtm.sh/',
        'exposedPort': PUBLIC_PORT,
    }


class FakeMaritime:
    """In-memory stand in for ``MaritimeClient``.

    ``statuses_after_create`` is the sequence of statuses a new agent reports
    on successive reads, ending on the last one.
    """

    def __init__(self, statuses_after_create: list[str] | None = None):
        self.agents: dict[str, dict[str, Any]] = {}
        self.statuses_after_create = statuses_after_create or ['active']
        self.created: list[dict[str, Any]] = []
        self.started: list[str] = []
        self.slept: list[str] = []
        self.deleted: list[str] = []
        self.list_calls: list[str | None] = []
        self.fail: dict[str, MaritimeApiError] = {}
        self._pending: list[str] = []

    def add(self, agent_id: str = AGENT_ID, status: str = 'active', **kwargs) -> None:
        self.agents[agent_id] = _agent(agent_id, status, **kwargs)

    def _maybe_fail(self, action: str) -> None:
        if action in self.fail:
            raise self.fail[action]

    async def create_agent(self, body: dict[str, Any]) -> dict[str, Any]:
        self._maybe_fail('create')
        self.created.append(body)
        self._pending = list(self.statuses_after_create)
        agent = _agent(AGENT_ID, 'deploying')
        self.agents[AGENT_ID] = agent
        return dict(agent)

    async def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        self._maybe_fail('get')
        agent = self.agents.get(agent_id)
        if agent is None:
            return None
        if agent_id == AGENT_ID and self._pending:
            agent['status'] = self._pending.pop(0)
        return dict(agent)

    async def list_agents(self, external_id: str | None = None) -> list[dict[str, Any]]:
        self._maybe_fail('list')
        self.list_calls.append(external_id)
        return [
            dict(agent)
            for agent in self.agents.values()
            if external_id is None or agent['externalId'] == external_id
        ]

    async def start_agent(self, agent_id: str) -> dict[str, Any] | None:
        self._maybe_fail('start')
        self.started.append(agent_id)
        self.agents[agent_id]['status'] = 'active'
        return dict(self.agents[agent_id])

    async def sleep_agent(self, agent_id: str) -> None:
        self._maybe_fail('sleep')
        self.slept.append(agent_id)
        self.agents[agent_id]['status'] = 'sleeping'

    async def delete_agent(self, agent_id: str) -> None:
        self._maybe_fail('delete')
        self.deleted.append(agent_id)
        self.agents.pop(agent_id, None)


def _response(status_code: int = 200, payload: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload if payload is not None else {},
        request=httpx.Request('GET', AGENT_URL),
    )


class FakeAgentServer:
    """httpx.AsyncClient stand in that answers the agent server routes.

    ``init_states`` scripts the answers to ``GET /api/init``. Once they run
    out, the server reports ``dormant`` until a ``POST /api/init`` claims it
    and ``ready`` after, starting claimed when ``claimed`` is set.
    """

    def __init__(
        self,
        init_states: list[Any] | None = None,
        init_post_status: int = 200,
        claimed: bool = False,
    ):
        self.init_states = init_states or []
        self.claimed = claimed
        self.init_post_status = init_post_status
        self.init_post_urls: list[str] = []
        self.init_post_bodies: list[dict] = []
        self.init_post_headers: list[dict] = []

    async def get(self, url: str, **kwargs):
        if url.endswith('/api/init'):
            if self.init_states:
                state = self.init_states.pop(0)
                if isinstance(state, Exception):
                    raise state
                if isinstance(state, int):
                    return _response(state)
                return _response(200, {'state': state, 'error': None})
            state = 'ready' if self.claimed else 'dormant'
            return _response(200, {'state': state, 'error': None})
        raise AssertionError(f'unexpected GET {url}')

    async def post(self, url: str, **kwargs):
        assert url.endswith('/api/init'), f'unexpected POST {url}'
        self.init_post_urls.append(url)
        self.init_post_bodies.append(kwargs['json'])
        self.init_post_headers.append(kwargs.get('headers') or {})
        if self.init_post_status == 200:
            self.claimed = True
        return _response(self.init_post_status, {'state': 'ready'})


def _stored(
    sandbox_id: str = AGENT_ID,
    created_by_user_id: str | None = OWNER_ID,
    session_api_key: str | None = SESSION_API_KEY,
) -> StoredSandbox:
    """The sandbox table row start_sandbox writes for a Maritime agent."""
    return StoredSandbox(
        id=sandbox_id,
        backend=MARITIME_BACKEND,
        created_by_user_id=created_by_user_id,
        sandbox_spec_id=IMAGE,
        session_api_key_hash=(
            hash_session_api_key(session_api_key) if session_api_key else None
        ),
        session_api_key=SecretStr(session_api_key) if session_api_key else None,
        created_at=CREATED_AT,
    )


def _user_context(user_id: str | None) -> AsyncMock:
    context = AsyncMock()
    context.get_user_id.return_value = user_id
    context.get_default_sandbox_spec_id.return_value = None
    return context


def _service(
    db_session,
    maritime: FakeMaritime | None = None,
    user_id: str | None = OWNER_ID,
    user_context: UserContext | None = None,
    httpx_client=None,
    web_url: str | None = WEB_URL,
    spec: MaritimeSandboxSpecInfo | None = None,
    init_timeout_seconds: int = 5,
) -> MaritimeSandboxService:
    spec = spec or MaritimeSandboxSpecInfo(
        id=IMAGE, command=None, working_dir='/workspace/project'
    )
    return MaritimeSandboxService(
        sandbox_spec_service=PresetSandboxSpecService(specs=[spec]),
        user_context=user_context or _user_context(user_id),
        httpx_client=httpx_client or FakeAgentServer(),
        db_session=db_session,
        maritime=maritime or FakeMaritime(),  # type: ignore[arg-type]
        agent_url_domain='mtm.sh',
        idle_ttl_seconds=43200,
        max_num_sandboxes=10,
        init_timeout_seconds=init_timeout_seconds,
        poll_interval=0,
        web_url=web_url,
        lifecycle=SandboxLifecycleSettings(),
    )


@pytest.fixture
async def db_session(async_session_maker):
    """A session on this test's own postgres database."""
    async with async_session_maker() as session:
        yield session


@pytest.fixture
def store(db_session):
    """Add the rows standing for the sandboxes a test sets up."""

    async def _store(*sandboxes: StoredSandbox) -> None:
        for sandbox in sandboxes:
            db_session.add(sandbox)
        await db_session.flush()

    return _store


async def _rows(db_session) -> list[StoredSandbox]:
    await db_session.flush()
    return list((await db_session.execute(select(StoredSandbox))).scalars())


# ---------------------------------------------------------------------------
# The HTTP client
# ---------------------------------------------------------------------------


def _client(handler) -> MaritimeClient:
    return MaritimeClient(
        httpx_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_key='mk_test',
        api_url=f'{API_URL}/',
    )


class TestMaritimeClient:
    @pytest.mark.asyncio
    async def test_create_posts_the_body_with_the_bearer_key(self):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(201, json=_agent(status='deploying'))

        agent = await _client(handler).create_agent({'name': 'openhands-x'})

        assert agent['id'] == AGENT_ID
        assert seen[0].method == 'POST'
        assert str(seen[0].url) == f'{API_URL}/api/agents'
        assert seen[0].headers['Authorization'] == 'Bearer mk_test'
        assert seen[0].read() == b'{"name":"openhands-x"}'

    @pytest.mark.asyncio
    async def test_list_filters_on_the_external_id(self):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=[_agent(), 'not-an-agent'])

        agents = await _client(handler).list_agents('openhands:user-1')

        assert seen[0].url.params['externalId'] == 'openhands:user-1'
        assert [agent['id'] for agent in agents] == [AGENT_ID]

    @pytest.mark.asyncio
    async def test_a_missing_agent_reads_as_none(self):
        client = _client(lambda _: httpx.Response(404, json={'detail': 'not found'}))

        assert await client.get_agent(AGENT_ID) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize('status_code', [401, 403])
    async def test_a_rejected_key_is_reported_as_itself(self, status_code):
        client = _client(lambda _: httpx.Response(status_code, json={'detail': 'no'}))

        with pytest.raises(MaritimeApiError) as exc_info:
            await client.get_agent(AGENT_ID)

        assert exc_info.value.status_code == status_code
        assert MARITIME_AUTH_FAILURE in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_an_error_carries_status_and_detail(self):
        client = _client(lambda _: httpx.Response(429, json={'detail': 'awake cap'}))

        with pytest.raises(MaritimeApiError) as exc_info:
            await client.start_agent(AGENT_ID)

        assert exc_info.value.status_code == 429
        assert 'awake cap' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_deleting_a_missing_agent_succeeds(self):
        client = _client(lambda _: httpx.Response(404, json={'detail': 'not found'}))

        await client.delete_agent(AGENT_ID)

    @pytest.mark.asyncio
    async def test_a_failed_delete_raises(self):
        client = _client(lambda _: httpx.Response(500, text='boom'))

        with pytest.raises(MaritimeApiError) as exc_info:
            await client.delete_agent(AGENT_ID)

        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_a_no_content_answer_returns_none(self):
        client = _client(lambda _: httpx.Response(204))

        assert await client.sleep_agent(AGENT_ID) is None

    @pytest.mark.asyncio
    async def test_a_transport_error_is_a_maritime_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError('down', request=request)

        with pytest.raises(MaritimeApiError, match='Could not reach Maritime'):
            await _client(handler).get_agent(AGENT_ID)


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


class TestStartSandbox:
    @pytest.mark.asyncio
    async def test_creates_an_agent_running_the_spec_image(self, db_session):
        maritime = FakeMaritime()

        sandbox = await _service(db_session, maritime).start_sandbox()

        body = maritime.created[0]
        assert body['framework'] == 'custom'
        assert body['imageName'] == IMAGE
        assert body['exposedPort'] == PUBLIC_PORT
        assert body['publicWeb'] is True
        assert body['externalId'] == f'{EXTERNAL_ID_PREFIX}{OWNER_ID}'
        assert body['name'].startswith('openhands-')
        assert body['memMb'] == 4096
        assert body['vcpus'] == 2.0
        assert body['idleTtlSeconds'] == 43200
        assert 'diskGb' not in body
        assert sandbox.id == AGENT_ID
        assert sandbox.status == SandboxStatus.RUNNING
        assert sandbox.sandbox_spec_id == IMAGE

    @pytest.mark.asyncio
    async def test_the_session_key_boots_the_server_dormant_as_a_secret(
        self, db_session
    ):
        maritime = FakeMaritime()

        sandbox = await _service(db_session, maritime).start_sandbox()

        env = {item['key']: item for item in maritime.created[0]['initialEnvVars']}
        assert env['OH_DEFERRED_INIT'] == {
            'key': 'OH_DEFERRED_INIT',
            'value': '1',
            'isSecret': False,
        }
        assert env['OH_SECRET_KEY']['value'] == sandbox.session_api_key
        assert env['OH_SECRET_KEY']['isSecret'] is True

    @pytest.mark.asyncio
    async def test_each_sandbox_gets_its_own_key_and_name(self, db_session):
        maritime = FakeMaritime()
        service = _service(db_session, maritime)

        first = await service.start_sandbox()
        await db_session.delete((await _rows(db_session))[0])
        service.httpx_client = FakeAgentServer()  # type: ignore[assignment]
        second = await service.start_sandbox()

        assert first.session_api_key != second.session_api_key
        assert maritime.created[0]['name'] != maritime.created[1]['name']

    @pytest.mark.asyncio
    async def test_disk_size_is_sent_when_the_spec_sets_one(self, db_session):
        maritime = FakeMaritime()
        spec = MaritimeSandboxSpecInfo(
            id=IMAGE, command=None, mem_mb=8192, vcpus=4, disk_gb=20
        )

        await _service(db_session, maritime, spec=spec).start_sandbox()

        body = maritime.created[0]
        assert (body['memMb'], body['vcpus'], body['diskGb']) == (8192, 4, 20)

    @pytest.mark.asyncio
    async def test_records_owner_spec_and_key_on_the_row(self, db_session):
        sandbox = await _service(db_session).start_sandbox()

        [row] = await _rows(db_session)
        assert row.backend == MARITIME_BACKEND
        assert row.id == AGENT_ID
        assert row.created_by_user_id == OWNER_ID
        assert row.sandbox_spec_id == IMAGE
        assert row.session_api_key.get_secret_value() == sandbox.session_api_key
        assert row.session_api_key_hash == hash_session_api_key(sandbox.session_api_key)

    @pytest.mark.asyncio
    async def test_waits_for_the_agent_to_deploy(self, db_session):
        maritime = FakeMaritime(
            statuses_after_create=['deploying', 'deploying', 'active']
        )

        sandbox = await _service(db_session, maritime).start_sandbox()

        assert sandbox.status == SandboxStatus.RUNNING
        assert maritime._pending == []

    @pytest.mark.asyncio
    async def test_exposes_only_the_agent_server_at_the_agent_url(self, db_session):
        sandbox = await _service(db_session).start_sandbox()

        assert [(url.name, url.url, url.port) for url in sandbox.exposed_urls] == [
            (AGENT_SERVER, AGENT_URL, PUBLIC_PORT)
        ]

    @pytest.mark.asyncio
    async def test_claims_the_agent_server_with_the_session_key(self, db_session):
        agent_server = FakeAgentServer(init_states=[503, 'dormant'])

        sandbox = await _service(db_session, httpx_client=agent_server).start_sandbox()

        key = sandbox.session_api_key
        assert agent_server.init_post_urls == [f'{AGENT_URL}/api/init']
        assert agent_server.init_post_headers == [{'X-Init-API-Key': key}]
        body = agent_server.init_post_bodies[0]
        assert body['session_api_keys'] == [key]
        assert body['secret_key'] == key
        assert body['conversations_path'] == '/workspace/conversations'
        assert body['bash_events_dir'] == '/workspace/bash_events'
        assert body['conversation_worktree_root'] == '/workspace/worktrees'
        assert body['webhooks'] == [{'base_url': f'{WEB_URL}/api/v1/webhooks'}]
        assert body['allow_cors_origins'] == [WEB_URL]

    @pytest.mark.asyncio
    async def test_init_env_carries_agent_server_env_lifecycle_and_extra_env(
        self, db_session, monkeypatch
    ):
        monkeypatch.setenv('LLM_TIMEOUT', '30')
        agent_server = FakeAgentServer()

        await _service(db_session, httpx_client=agent_server).start_sandbox(
            extra_env={'GIT_AUTHOR': 'ada'}
        )

        env = agent_server.init_post_bodies[0]['env']
        assert env['LLM_TIMEOUT'] == '30'
        assert env['GIT_AUTHOR'] == 'ada'
        assert env['OH_RUNTIME_IDLE_TIMEOUT_SECONDS'] == '1200'
        # Maritime publishes one port, so the worker ports are not offered.
        assert 'WORKER_1' not in env

    @pytest.mark.asyncio
    async def test_no_webhooks_without_a_public_web_url(self, db_session):
        agent_server = FakeAgentServer()

        await _service(
            db_session, httpx_client=agent_server, web_url='http://localhost:3000'
        ).start_sandbox()

        assert 'webhooks' not in agent_server.init_post_bodies[0]

    @pytest.mark.asyncio
    async def test_needs_an_owner(self, db_session):
        maritime = FakeMaritime()
        service = _service(db_session, maritime, user_context=ADMIN)

        with (
            patch.object(service, 'pause_old_sandboxes') as pause,
            pytest.raises(AuthError),
        ):
            await service.start_sandbox()

        pause.assert_not_called()
        assert maritime.created == []

    @pytest.mark.asyncio
    async def test_a_full_plan_says_so_and_writes_nothing(self, db_session):
        maritime = FakeMaritime()
        maritime.fail['create'] = MaritimeApiError('awake cap', 429)

        with pytest.raises(SandboxError, match='no room for another awake agent'):
            await _service(db_session, maritime).start_sandbox()

        assert await _rows(db_session) == []

    @pytest.mark.asyncio
    async def test_an_agent_that_fails_to_deploy_is_deleted(self, db_session):
        maritime = FakeMaritime(statuses_after_create=['deploying', 'error'])

        with pytest.raises(SandboxError, match='failed to start'):
            await _service(db_session, maritime).start_sandbox()

        assert maritime.deleted == [AGENT_ID]

    @pytest.mark.asyncio
    async def test_an_agent_that_never_deploys_times_out_and_is_deleted(
        self, db_session
    ):
        maritime = FakeMaritime(statuses_after_create=['deploying'] * 1000)

        with pytest.raises(SandboxError, match='did not start within 0s'):
            await _service(db_session, maritime, init_timeout_seconds=0).start_sandbox()

        assert maritime.deleted == [AGENT_ID]

    @pytest.mark.asyncio
    async def test_a_rejected_init_key_is_explained_and_the_agent_deleted(
        self, db_session
    ):
        maritime = FakeMaritime()
        agent_server = FakeAgentServer(init_post_status=401)

        with pytest.raises(SandboxError, match='rejected the init API key'):
            await _service(
                db_session, maritime, httpx_client=agent_server
            ).start_sandbox()

        assert maritime.deleted == [AGENT_ID]

    @pytest.mark.asyncio
    async def test_an_initialized_server_is_not_claimed_twice(self, db_session):
        maritime = FakeMaritime()
        agent_server = FakeAgentServer(init_states=['ready'])

        with pytest.raises(SandboxError, match='already initialized'):
            await _service(
                db_session, maritime, httpx_client=agent_server
            ).start_sandbox()

        assert agent_server.init_post_bodies == []
        assert maritime.deleted == [AGENT_ID]

    @pytest.mark.asyncio
    async def test_a_failed_cleanup_keeps_the_original_error(self, db_session):
        maritime = FakeMaritime(statuses_after_create=['error'])
        maritime.fail['delete'] = MaritimeApiError('down', 503)

        with pytest.raises(SandboxError, match='failed to start'):
            await _service(db_session, maritime).start_sandbox()


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


class TestReads:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'maritime_status, expected',
        [
            ('active', SandboxStatus.RUNNING),
            ('sleeping', SandboxStatus.PAUSED),
            ('stopped', SandboxStatus.PAUSED),
            ('deploying', SandboxStatus.STARTING),
            ('error', SandboxStatus.ERROR),
            ('something-new', SandboxStatus.ERROR),
        ],
    )
    async def test_status_mapping(self, db_session, store, maritime_status, expected):
        maritime = FakeMaritime()
        maritime.add(status=maritime_status)
        await store(_stored())

        sandbox = await _service(db_session, maritime).get_sandbox(AGENT_ID)

        assert sandbox.status == expected

    @pytest.mark.asyncio
    async def test_a_row_maritime_no_longer_has_is_missing(self, db_session, store):
        await store(_stored())

        sandbox = await _service(db_session, FakeMaritime()).get_sandbox(AGENT_ID)

        assert sandbox.status == SandboxStatus.MISSING
        assert sandbox.session_api_key is None
        assert sandbox.exposed_urls is None

    @pytest.mark.asyncio
    async def test_key_and_url_only_while_running(self, db_session, store):
        maritime = FakeMaritime()
        maritime.add(status='sleeping')
        await store(_stored())
        service = _service(db_session, maritime)

        paused = await service.get_sandbox(AGENT_ID)
        maritime.agents[AGENT_ID]['status'] = 'active'
        running = await service.get_sandbox(AGENT_ID)

        assert (paused.session_api_key, paused.exposed_urls) == (None, None)
        assert running.session_api_key == SESSION_API_KEY
        assert running.exposed_urls[0].url == AGENT_URL

    @pytest.mark.asyncio
    async def test_url_falls_back_to_the_agent_url_domain(self, db_session, store):
        maritime = FakeMaritime()
        maritime.add()
        maritime.agents[AGENT_ID]['webUrl'] = None
        await store(_stored())

        sandbox = await _service(db_session, maritime).get_sandbox(AGENT_ID)

        assert sandbox.exposed_urls[0].url == AGENT_URL

    @pytest.mark.asyncio
    async def test_another_users_sandbox_is_invisible(self, db_session, store):
        maritime = FakeMaritime()
        maritime.add()
        await store(_stored())

        service = _service(db_session, maritime, user_id=OTHER_USER_ID)

        assert await service.get_sandbox(AGENT_ID) is None
        assert (await service.search_sandboxes()).items == []

    @pytest.mark.asyncio
    async def test_search_lists_the_users_agents_once(self, db_session, store):
        maritime = FakeMaritime()
        maritime.add()
        maritime.add(OTHER_AGENT_ID, status='sleeping')
        await store(_stored(), _stored(OTHER_AGENT_ID, session_api_key='other-key'))

        page = await _service(db_session, maritime).search_sandboxes()

        assert {item.id: item.status for item in page.items} == {
            AGENT_ID: SandboxStatus.RUNNING,
            OTHER_AGENT_ID: SandboxStatus.PAUSED,
        }
        assert maritime.list_calls == [f'{EXTERNAL_ID_PREFIX}{OWNER_ID}']

    @pytest.mark.asyncio
    async def test_admin_search_lists_without_the_owner_filter(self, db_session, store):
        maritime = FakeMaritime()
        maritime.add(user_id=OTHER_USER_ID)
        await store(_stored(created_by_user_id=OTHER_USER_ID))

        page = await _service(
            db_session, maritime, user_context=ADMIN
        ).search_sandboxes()

        assert [item.id for item in page.items] == [AGENT_ID]
        assert maritime.list_calls == [None]

    @pytest.mark.asyncio
    async def test_an_empty_page_makes_no_maritime_call(self, db_session):
        maritime = FakeMaritime()

        page = await _service(db_session, maritime).search_sandboxes()

        assert page.items == []
        assert maritime.list_calls == []

    @pytest.mark.asyncio
    async def test_lookup_by_session_key(self, db_session, store):
        maritime = FakeMaritime()
        maritime.add()
        await store(_stored())
        service = _service(db_session, maritime)

        sandbox = await service.get_sandbox_by_session_api_key(SESSION_API_KEY)
        record = await service.get_sandbox_record_by_session_api_key(SESSION_API_KEY)

        assert sandbox.id == AGENT_ID
        assert sandbox.status == SandboxStatus.RUNNING
        assert (record.id, record.created_by_user_id) == (AGENT_ID, OWNER_ID)
        assert await service.get_sandbox_by_session_api_key('wrong') is None

    @pytest.mark.asyncio
    async def test_record_lookup_makes_no_maritime_call(self, db_session, store):
        maritime = FakeMaritime()
        maritime.fail['get'] = MaritimeApiError('must not be called', 500)
        await store(_stored())

        record = await _service(
            db_session, maritime
        ).get_sandbox_record_by_session_api_key(SESSION_API_KEY)

        assert record.id == AGENT_ID


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestPause:
    @pytest.mark.asyncio
    async def test_puts_a_running_agent_to_sleep(self, db_session, store):
        maritime = FakeMaritime()
        maritime.add()
        await store(_stored())

        assert await _service(db_session, maritime).pause_sandbox(AGENT_ID) is True
        assert maritime.slept == [AGENT_ID]

    @pytest.mark.asyncio
    @pytest.mark.parametrize('status', ['sleeping', 'stopped'])
    async def test_an_asleep_agent_is_left_alone(self, db_session, store, status):
        maritime = FakeMaritime()
        maritime.add(status=status)
        await store(_stored())

        assert await _service(db_session, maritime).pause_sandbox(AGENT_ID) is True
        assert maritime.slept == []

    @pytest.mark.asyncio
    async def test_a_deploying_agent_is_skipped(self, db_session, store):
        maritime = FakeMaritime()
        maritime.add(status='deploying')
        await store(_stored())

        assert await _service(db_session, maritime).pause_sandbox(AGENT_ID) is True
        assert maritime.slept == []

    @pytest.mark.asyncio
    async def test_a_missing_agent_cannot_be_paused(self, db_session, store):
        await store(_stored())

        assert (
            await _service(db_session, FakeMaritime()).pause_sandbox(AGENT_ID) is False
        )

    @pytest.mark.asyncio
    async def test_a_refused_sleep_fails(self, db_session, store):
        maritime = FakeMaritime()
        maritime.add()
        maritime.fail['sleep'] = MaritimeApiError('boom', 500)
        await store(_stored())

        assert await _service(db_session, maritime).pause_sandbox(AGENT_ID) is False


class TestResume:
    @pytest.mark.asyncio
    async def test_wakes_a_sleeping_agent_without_a_new_handshake(
        self, db_session, store
    ):
        maritime = FakeMaritime()
        maritime.add(status='sleeping')
        agent_server = FakeAgentServer(init_states=['ready'])
        await store(_stored())

        service = _service(db_session, maritime, httpx_client=agent_server)

        assert await service.resume_sandbox(AGENT_ID) is True
        assert maritime.started == [AGENT_ID]
        assert agent_server.init_post_bodies == []

    @pytest.mark.asyncio
    async def test_a_cold_restart_is_claimed_again_with_the_stored_key(
        self, db_session, store
    ):
        maritime = FakeMaritime()
        maritime.add(status='stopped')
        agent_server = FakeAgentServer(init_states=[503, 'dormant'])
        await store(_stored())

        service = _service(db_session, maritime, httpx_client=agent_server)

        assert await service.resume_sandbox(AGENT_ID) is True
        assert agent_server.init_post_headers == [{'X-Init-API-Key': SESSION_API_KEY}]
        body = agent_server.init_post_bodies[0]
        assert body['session_api_keys'] == [SESSION_API_KEY]
        # The same secret key, so secrets persisted before the restart decrypt.
        assert body['secret_key'] == SESSION_API_KEY

    @pytest.mark.asyncio
    async def test_a_running_agent_is_not_started_again(self, db_session, store):
        maritime = FakeMaritime()
        maritime.add()
        agent_server = FakeAgentServer(init_states=['ready'])
        await store(_stored())

        service = _service(db_session, maritime, httpx_client=agent_server)

        assert await service.resume_sandbox(AGENT_ID) is True
        assert maritime.started == []

    @pytest.mark.asyncio
    async def test_a_full_plan_says_so(self, db_session, store):
        maritime = FakeMaritime()
        maritime.add(status='sleeping')
        maritime.fail['start'] = MaritimeApiError('awake cap', 429)
        await store(_stored())

        with pytest.raises(SandboxError, match='no room for another awake agent'):
            await _service(db_session, maritime).resume_sandbox(AGENT_ID)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('status', ['error', None])
    async def test_a_broken_or_missing_agent_cannot_be_resumed(
        self, db_session, store, status
    ):
        maritime = FakeMaritime()
        if status:
            maritime.add(status=status)
        await store(_stored())

        assert await _service(db_session, maritime).resume_sandbox(AGENT_ID) is False
        assert maritime.started == []

    @pytest.mark.asyncio
    async def test_a_refused_start_fails(self, db_session, store):
        maritime = FakeMaritime()
        maritime.add(status='sleeping')
        maritime.fail['start'] = MaritimeApiError('boom', 500)
        await store(_stored())

        assert await _service(db_session, maritime).resume_sandbox(AGENT_ID) is False


class TestDelete:
    @pytest.mark.asyncio
    async def test_deletes_the_agent_and_the_row(self, db_session, store):
        maritime = FakeMaritime()
        maritime.add()
        await store(_stored())

        assert await _service(db_session, maritime).delete_sandbox(AGENT_ID) is True
        assert maritime.deleted == [AGENT_ID]
        assert await _rows(db_session) == []

    @pytest.mark.asyncio
    async def test_a_failed_delete_keeps_the_row_for_a_retry(self, db_session, store):
        maritime = FakeMaritime()
        maritime.add()
        maritime.fail['delete'] = MaritimeApiError('boom', 500)
        await store(_stored())

        with pytest.raises(SandboxDeleteRetryError):
            await _service(db_session, maritime).delete_sandbox(AGENT_ID)

        assert [row.id for row in await _rows(db_session)] == [AGENT_ID]

    @pytest.mark.asyncio
    async def test_another_users_sandbox_cannot_be_deleted(self, db_session, store):
        maritime = FakeMaritime()
        maritime.add()
        await store(_stored())

        service = _service(db_session, maritime, user_id=OTHER_USER_ID)

        assert await service.delete_sandbox(AGENT_ID) is False
        assert maritime.deleted == []


# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------


class TestInjector:
    def test_reads_its_settings_from_env(self, monkeypatch):
        monkeypatch.setenv('MARITIME_API_KEY', 'mk_env')
        monkeypatch.setenv('MARITIME_API_URL', 'https://api.example.com')
        monkeypatch.setenv('MARITIME_IDLE_TTL_SECONDS', '600')

        injector = MaritimeSandboxServiceInjector()

        assert injector.api_key == 'mk_env'
        assert injector.api_url == 'https://api.example.com'
        assert injector._effective_idle_ttl() == 600

    def test_idle_ttl_defaults_to_the_lifecycle_max_session(self, monkeypatch):
        monkeypatch.delenv('MARITIME_IDLE_TTL_SECONDS', raising=False)

        injector = MaritimeSandboxServiceInjector(
            lifecycle=SandboxLifecycleSettings(max_session_seconds=7200)
        )

        assert injector._effective_idle_ttl() == 7200

    def test_idle_ttl_has_a_backstop_when_max_session_is_off(self, monkeypatch):
        monkeypatch.delenv('MARITIME_IDLE_TTL_SECONDS', raising=False)

        injector = MaritimeSandboxServiceInjector(
            lifecycle=SandboxLifecycleSettings(max_session_seconds=0)
        )

        assert injector._effective_idle_ttl() == 43200

    def test_runtime_maritime_selects_the_maritime_injectors(self, monkeypatch):
        from openhands.app_server import config as config_module
        from openhands.app_server.sandbox.maritime_sandbox_spec_service import (
            MaritimeSandboxSpecServiceInjector,
        )

        monkeypatch.setenv('RUNTIME', 'maritime')
        config = config_module.config_from_env()

        assert isinstance(config.sandbox, MaritimeSandboxServiceInjector)
        assert isinstance(config.sandbox_spec, MaritimeSandboxSpecServiceInjector)
