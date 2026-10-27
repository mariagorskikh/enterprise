"""Tests for the Maritime sandbox spec service.

Covers the defaults the backend ships with: the published Maritime image that
matches the installed agent server, and the size settings read from env.
"""

from importlib.metadata import version
from unittest.mock import MagicMock

import pytest

from openhands.app_server.sandbox import maritime_sandbox_spec_service
from openhands.app_server.sandbox.maritime_sandbox_spec_service import (
    DEFAULT_IMAGE_REPOSITORY,
    DEFAULT_MEM_MB,
    DEFAULT_VCPUS,
    DEFAULT_WORKING_DIR,
    MaritimeSandboxSpecInfo,
    MaritimeSandboxSpecServiceInjector,
    default_image,
    get_default_sandbox_specs,
)
from openhands.app_server.sandbox.preset_sandbox_spec_service import (
    PresetSandboxSpecService,
)

ENV_VARS = ('MARITIME_IMAGE', 'MARITIME_MEM_MB', 'MARITIME_VCPUS', 'MARITIME_DISK_GB')


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


class TestDefaults:
    def test_default_image_matches_the_installed_agent_server(self):
        assert default_image() == (
            f'{DEFAULT_IMAGE_REPOSITORY}:{version("openhands-agent-server")}'
        )

    def test_default_image_without_the_agent_server_package(self, monkeypatch):
        def missing(name):
            raise maritime_sandbox_spec_service.PackageNotFoundError(name)

        monkeypatch.setattr(maritime_sandbox_spec_service, 'version', missing)

        assert default_image() == f'{DEFAULT_IMAGE_REPOSITORY}:latest'

    def test_default_spec(self, clean_env):
        [spec] = get_default_sandbox_specs()

        assert spec.id == default_image()
        assert spec.command is None
        assert spec.working_dir == DEFAULT_WORKING_DIR
        assert spec.mem_mb == DEFAULT_MEM_MB
        assert spec.vcpus == DEFAULT_VCPUS
        assert spec.disk_gb is None

    def test_default_spec_from_env(self, clean_env):
        clean_env.setenv('MARITIME_IMAGE', 'ghcr.io/acme/agent-server:custom')
        clean_env.setenv('MARITIME_MEM_MB', '8192')
        clean_env.setenv('MARITIME_VCPUS', '4')
        clean_env.setenv('MARITIME_DISK_GB', '20')

        [spec] = get_default_sandbox_specs()

        assert spec.id == 'ghcr.io/acme/agent-server:custom'
        assert (spec.mem_mb, spec.vcpus, spec.disk_gb) == (8192, 4.0, 20.0)

    def test_initial_env_is_empty(self, clean_env):
        """Nothing is baked into the spec: it is published verbatim."""
        clean_env.setenv('LLM_API_KEY', 'sk-do-not-publish-this')

        assert get_default_sandbox_specs()[0].initial_env == {}


class TestInjector:
    @pytest.mark.asyncio
    async def test_yields_preset_service_with_specs(self):
        spec = MaritimeSandboxSpecInfo(id='ghcr.io/acme/image:1', command=None)
        injector = MaritimeSandboxSpecServiceInjector(specs=[spec])

        async with injector.context(MagicMock()) as service:
            assert isinstance(service, PresetSandboxSpecService)
            assert await service.get_sandbox_spec('ghcr.io/acme/image:1') == spec
            assert await service.get_default_sandbox_spec() == spec

    @pytest.mark.asyncio
    async def test_yields_default_specs(self, clean_env):
        async with MaritimeSandboxSpecServiceInjector().context(MagicMock()) as service:
            page = await service.search_sandbox_specs()

        assert [item.id for item in page.items] == [default_image()]
