import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "frontend" / "core" / "gpu_utils.py"
MODULE_SPEC = importlib.util.spec_from_file_location("frontend_core_gpu_utils", MODULE_PATH)
gpu_utils = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(gpu_utils)


@pytest.mark.parametrize(
    ("raw_gpu_devices", "physical_gpu_count", "expected"),
    [
        (None, 1, "0"),
        ("0", 1, "0"),
        (" 0 , 1 ", 2, "0,1"),
    ],
)
def test_normalize_requested_gpu_devices_accepts_valid_ids(
    monkeypatch: pytest.MonkeyPatch,
    raw_gpu_devices: str | None,
    physical_gpu_count: int,
    expected: str,
):
    monkeypatch.setattr(
        gpu_utils, "get_physical_gpu_count", lambda: physical_gpu_count
    )

    assert gpu_utils.normalize_requested_gpu_devices(raw_gpu_devices) == expected


@pytest.mark.parametrize(
    ("raw_gpu_devices", "physical_gpu_count", "expected_message"),
    [
        ("4", 1, r"This machine has 1 physical GPU\(s\)"),
        ("0,,1", 2, "comma-separated list of integers"),
        ("a", 2, "Invalid GPU device ID 'a'"),
        ("-1", 2, "must be non-negative integers"),
        ("0,0", 2, "Duplicate GPU device ID '0'"),
    ],
)
def test_normalize_requested_gpu_devices_rejects_invalid_ids(
    monkeypatch: pytest.MonkeyPatch,
    raw_gpu_devices: str,
    physical_gpu_count: int,
    expected_message: str,
):
    monkeypatch.setattr(
        gpu_utils, "get_physical_gpu_count", lambda: physical_gpu_count
    )

    with pytest.raises(ValueError, match=expected_message):
        gpu_utils.normalize_requested_gpu_devices(raw_gpu_devices)
