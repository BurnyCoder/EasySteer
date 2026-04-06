import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
frontend_pkg = types.ModuleType("frontend")
frontend_pkg.__path__ = [str(ROOT / "frontend")]
sys.modules.setdefault("frontend", frontend_pkg)

core_pkg = types.ModuleType("frontend.core")
core_pkg.__path__ = [str(ROOT / "frontend" / "core")]
sys.modules.setdefault("frontend.core", core_pkg)

vllm_stub = types.ModuleType("vllm")
vllm_stub.LLM = object
sys.modules.setdefault("vllm", vllm_stub)

MODULE_PATH = ROOT / "frontend" / "core" / "llm_manager.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "frontend.core.llm_manager",
    MODULE_PATH,
)
llm_manager_module = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
sys.modules[MODULE_SPEC.name] = llm_manager_module
MODULE_SPEC.loader.exec_module(llm_manager_module)


def make_manager_with_stubbed_llm(monkeypatch: pytest.MonkeyPatch):
    created_configs = []

    def fake_llm(**kwargs):
        config = dict(kwargs)
        created_configs.append(config)
        return {"instance": len(created_configs), "config": config}

    monkeypatch.setattr(
        llm_manager_module,
        "normalize_requested_gpu_devices",
        lambda gpu_devices: "0" if gpu_devices is None else str(gpu_devices).strip(),
    )
    monkeypatch.setattr(llm_manager_module, "LLM", fake_llm)

    return llm_manager_module.LLMManager(), created_configs


def test_steering_without_prefix_override_disables_prefix_caching(
    monkeypatch: pytest.MonkeyPatch,
):
    manager, created_configs = make_manager_with_stubbed_llm(monkeypatch)

    manager.get_or_create_llm("demo-model", enable_steer_vector=True)

    assert created_configs[0]["enable_steer_vector"] is True
    assert created_configs[0]["enable_prefix_caching"] is False


def test_steering_rejects_explicit_prefix_caching(
    monkeypatch: pytest.MonkeyPatch,
):
    manager, created_configs = make_manager_with_stubbed_llm(monkeypatch)

    with pytest.raises(
        ValueError,
        match="Steer vectors are incompatible with prefix caching",
    ):
        manager.get_or_create_llm(
            "demo-model",
            enable_steer_vector=True,
            enable_prefix_caching=True,
        )

    assert created_configs == []


def test_non_steering_does_not_force_prefix_caching_override(
    monkeypatch: pytest.MonkeyPatch,
):
    manager, created_configs = make_manager_with_stubbed_llm(monkeypatch)

    manager.get_or_create_llm("demo-model", enable_steer_vector=False)

    assert "enable_prefix_caching" not in created_configs[0]


def test_cache_key_distinguishes_non_steering_prefix_caching_modes(
    monkeypatch: pytest.MonkeyPatch,
):
    manager, created_configs = make_manager_with_stubbed_llm(monkeypatch)

    first = manager.get_or_create_llm("demo-model", enable_steer_vector=False)
    second = manager.get_or_create_llm(
        "demo-model",
        enable_steer_vector=False,
        enable_prefix_caching=False,
    )

    assert first != second
    assert len(created_configs) == 2


def test_steering_default_and_explicit_false_share_same_cached_instance(
    monkeypatch: pytest.MonkeyPatch,
):
    manager, created_configs = make_manager_with_stubbed_llm(monkeypatch)

    first = manager.get_or_create_llm("demo-model", enable_steer_vector=True)
    second = manager.get_or_create_llm(
        "demo-model",
        enable_steer_vector=True,
        enable_prefix_caching=False,
    )

    assert first == second
    assert len(created_configs) == 1
