import builtins
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def install_stubbed_runtime(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, tuple, dict]],
):
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = FakeCuda

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []

    class FakeSamplingParams:
        def __init__(self, *args, **kwargs):
            calls.append(("SamplingParams", args, kwargs))

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            calls.append(("LLM.__init__", args, kwargs))

        def generate(self, *args, **kwargs):
            calls.append(("LLM.generate", args, kwargs))
            return [
                types.SimpleNamespace(
                    outputs=[types.SimpleNamespace(text="stub output")]
                )
            ]

    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams

    fake_vllm_steer_vectors = types.ModuleType("vllm.steer_vectors")
    fake_vllm_steer_vectors.__path__ = []

    fake_request_module = types.ModuleType("vllm.steer_vectors.request")

    class FakeSteerVectorRequest:
        def __init__(self, *args, **kwargs):
            calls.append(("SteerVectorRequest", args, kwargs))

    fake_request_module.SteerVectorRequest = FakeSteerVectorRequest
    fake_vllm.steer_vectors = fake_vllm_steer_vectors
    fake_vllm_steer_vectors.request = fake_request_module

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.steer_vectors", fake_vllm_steer_vectors)
    monkeypatch.setitem(sys.modules, "vllm.steer_vectors.request", fake_request_module)


def load_module(module_name: str, relative_path: str):
    module_path = ROOT / relative_path
    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("module_name", "relative_path"),
    [
        ("easysteer_steering_demo", "easysteer-steering.py"),
        ("docker_test_demo", "docker/docker_test.py"),
    ],
)
def test_demo_scripts_are_import_safe(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    relative_path: str,
):
    calls: list[tuple[str, tuple, dict]] = []
    install_stubbed_runtime(monkeypatch, calls)

    module = load_module(module_name, relative_path)

    assert hasattr(module, "main")
    assert calls == []


@pytest.mark.parametrize(
    ("module_name", "relative_path", "expected_model"),
    [
        (
            "easysteer_steering_demo_main",
            "easysteer-steering.py",
            "Qwen/Qwen2.5-1.5B-Instruct",
        ),
        (
            "docker_test_demo_main",
            "docker/docker_test.py",
            "/app/models/Qwen/Qwen2.5-1.5B-Instruct/",
        ),
    ],
)
def test_demo_scripts_main_executes_with_stubbed_runtime(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    relative_path: str,
    expected_model: str,
):
    calls: list[tuple[str, tuple, dict]] = []
    install_stubbed_runtime(monkeypatch, calls)
    printed_lines = []
    monkeypatch.setattr(
        builtins,
        "print",
        lambda *args, **kwargs: printed_lines.append(args),
    )

    module = load_module(module_name, relative_path)
    module.main()

    llm_init_calls = [call for call in calls if call[0] == "LLM.__init__"]
    generate_calls = [call for call in calls if call[0] == "LLM.generate"]
    request_calls = [call for call in calls if call[0] == "SteerVectorRequest"]

    assert len(llm_init_calls) == 1
    assert llm_init_calls[0][2]["model"] == expected_model
    assert llm_init_calls[0][2]["enable_steer_vector"] is True
    assert llm_init_calls[0][2]["enable_prefix_caching"] is False
    assert len(request_calls) == 2
    assert len(generate_calls) == 2
    assert printed_lines == [("stub output",), ("stub output",)]
