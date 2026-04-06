"""
Helpers for validating GPU selections before mutating CUDA visibility.
"""

from __future__ import annotations

import logging
from functools import cache

logger = logging.getLogger(__name__)


@cache
def get_physical_gpu_count() -> int:
    """Return the physical NVIDIA GPU count without relying on CUDA visibility."""
    try:
        import vllm.third_party.pynvml as pynvml

        pynvml.nvmlInit()
        try:
            return int(pynvml.nvmlDeviceGetCount())
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # pragma: no cover - fallback only
        logger.warning("Falling back to torch.cuda.device_count(): %s", exc)

    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception as exc:  # pragma: no cover - fallback only
        logger.warning("Unable to detect GPU count via torch: %s", exc)
        return 0


def normalize_requested_gpu_devices(gpu_devices: str | None, default: str = "0") -> str:
    """Validate and normalize a comma-separated list of physical GPU ids."""
    requested = default if gpu_devices is None else str(gpu_devices).strip()
    if not requested:
        raise ValueError("GPU device IDs cannot be empty. Use a value like '0' or '0,1'.")

    tokens = [token.strip() for token in requested.split(",")]
    if any(token == "" for token in tokens):
        raise ValueError(
            "GPU device IDs must be a comma-separated list of integers like '0' or '0,1'."
        )

    physical_gpu_count = get_physical_gpu_count()
    if physical_gpu_count <= 0:
        raise ValueError(
            "No NVIDIA GPUs were detected. EasySteer requires at least one GPU."
        )

    parsed_ids: list[int] = []
    seen_ids: set[int] = set()
    for token in tokens:
        try:
            gpu_id = int(token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid GPU device ID '{token}'. Use non-negative integers like '0' or '0,1'."
            ) from exc

        if gpu_id < 0:
            raise ValueError(
                f"Invalid GPU device ID '{token}'. GPU IDs must be non-negative integers."
            )

        if gpu_id in seen_ids:
            raise ValueError(
                f"Duplicate GPU device ID '{gpu_id}' is not allowed in '{requested}'."
            )

        if gpu_id >= physical_gpu_count:
            raise ValueError(
                f"Invalid GPU device ID '{gpu_id}'. This machine has "
                f"{physical_gpu_count} physical GPU(s), so valid IDs are 0-{physical_gpu_count - 1}."
            )

        seen_ids.add(gpu_id)
        parsed_ids.append(gpu_id)

    return ",".join(str(gpu_id) for gpu_id in parsed_ids)
