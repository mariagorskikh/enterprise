import os
from importlib.metadata import PackageNotFoundError, version
from typing import AsyncGenerator

from fastapi import Request
from pydantic import Field

from openhands.app_server.sandbox.preset_sandbox_spec_service import (
    PresetSandboxSpecService,
)
from openhands.app_server.sandbox.sandbox_spec_models import SandboxSpecInfo
from openhands.app_server.sandbox.sandbox_spec_service import (
    SandboxSpecService,
    SandboxSpecServiceInjector,
)
from openhands.app_server.services.injector import InjectorState

DEFAULT_IMAGE_REPOSITORY = 'ghcr.io/maritime-sh/openhands-agent-server'
DEFAULT_WORKING_DIR = '/workspace/project'
DEFAULT_MEM_MB = 4096
DEFAULT_VCPUS = 2.0


class MaritimeSandboxSpecInfo(SandboxSpecInfo):
    """A sandbox spec whose ``id`` is the image a Maritime agent runs.

    The image is the published agent server with a launcher that binds the
    port Maritime routes to (see github.com/maritime-sh/openhands-agent-server).
    The Settings dropdown labels each option with ``spec.id``, so images appear
    there with no frontend change.

    ``initial_env`` must hold only values that are safe to publish: the public
    ``GET /api/v1/sandbox-specs/search`` endpoint serializes it verbatim.
    Credentials belong in the ``/api/init`` body that ``MaritimeSandboxService``
    builds per sandbox.
    """

    mem_mb: int = Field(
        default=DEFAULT_MEM_MB,
        description="Memory for each agent's micro-VM, in MB",
    )
    vcpus: float = Field(
        default=DEFAULT_VCPUS,
        description="vCPUs for each agent's micro-VM",
    )
    disk_gb: float | None = Field(
        default=None,
        description=(
            'Size of the persistent /data volume, in GB. None keeps the size '
            'the Maritime plan gives each agent.'
        ),
    )


def default_image() -> str:
    """The Maritime image that matches the installed agent server."""
    try:
        return f'{DEFAULT_IMAGE_REPOSITORY}:{version("openhands-agent-server")}'
    except PackageNotFoundError:
        return f'{DEFAULT_IMAGE_REPOSITORY}:latest'


def _env_number(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def get_default_sandbox_specs() -> list[MaritimeSandboxSpecInfo]:
    """The single image the Maritime backend ships with.

    ``command`` is None because the image carries its own entrypoint.
    """
    disk_gb = os.getenv('MARITIME_DISK_GB')
    return [
        MaritimeSandboxSpecInfo(
            id=os.getenv('MARITIME_IMAGE') or default_image(),
            command=None,
            working_dir=DEFAULT_WORKING_DIR,
            mem_mb=int(_env_number('MARITIME_MEM_MB', DEFAULT_MEM_MB)),
            vcpus=_env_number('MARITIME_VCPUS', DEFAULT_VCPUS),
            disk_gb=float(disk_gb) if disk_gb else None,
        )
    ]


class MaritimeSandboxSpecServiceInjector(SandboxSpecServiceInjector):
    """Dependency injector for Maritime sandbox spec services."""

    specs: list[MaritimeSandboxSpecInfo] = Field(
        default_factory=get_default_sandbox_specs,
        description='Preset list of images offered as Maritime sandbox specs',
    )

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[SandboxSpecService, None]:
        specs: list[SandboxSpecInfo] = list(self.specs)
        yield PresetSandboxSpecService(specs=specs)
