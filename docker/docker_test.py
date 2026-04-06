import multiprocessing
import os

import torch
from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest


def require_visible_cuda_device() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise SystemExit(
            "No CUDA GPUs are visible. Set CUDA_VISIBLE_DEVICES to a valid GPU "
            "index before running this demo, for example: "
            "CUDA_VISIBLE_DEVICES=0 python docker/docker_test.py"
        )


def resolve_gpu_memory_utilization() -> float:
    override = os.getenv("EASYSTEER_GPU_MEMORY_UTILIZATION")
    if override is not None:
        try:
            value = float(override)
        except ValueError as exc:
            raise SystemExit(
                "EASYSTEER_GPU_MEMORY_UTILIZATION must be a float between 0 and 1."
            ) from exc
        if not 0 < value <= 1:
            raise SystemExit(
                "EASYSTEER_GPU_MEMORY_UTILIZATION must be between 0 and 1."
            )
        return value

    mem_get_info = getattr(torch.cuda, "mem_get_info", None)
    if mem_get_info is None:
        return 0.8

    free_memory, total_memory = mem_get_info()
    safe_utilization = (free_memory / total_memory) * 0.95
    return max(0.05, min(0.8, safe_utilization))


def resolve_max_model_len() -> int:
    try:
        value = int(os.getenv("EASYSTEER_MAX_MODEL_LEN", "-1"))
    except ValueError as exc:
        raise SystemExit(
            "EASYSTEER_MAX_MODEL_LEN must be -1 or a positive integer."
        ) from exc
    if value == 0 or value < -1:
        raise SystemExit("EASYSTEER_MAX_MODEL_LEN must be -1 or a positive integer.")
    return value


def main() -> None:
    require_visible_cuda_device()
    gpu_memory_utilization = resolve_gpu_memory_utilization()
    max_model_len = resolve_max_model_len()

    # Initialize the LLM model.
    # enable_steer_vector=True: Enables vector steering.
    # enforce_eager=True: Ensures reliable interventions.
    # enable_chunked_prefill=False: Avoids known issues for this demo.
    # enable_prefix_caching=False: Prefix caching is incompatible with steering.
    # gpu_memory_utilization/max_model_len: Use safer defaults for smaller GPUs.
    llm = LLM(
        model="/app/models/Qwen/Qwen2.5-1.5B-Instruct/",
        enable_steer_vector=True,
        enforce_eager=True,
        tensor_parallel_size=1,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
    )
    text = (
        "<|im_start|>user\nAlice's dog has passed away. Please comfort her."
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    target_layers = list(range(10, 26))

    baseline_request = SteerVectorRequest(
        "baseline",
        1,
        steer_vector_local_path="/app/easysteer/vectors/happy_diffmean.gguf",
        scale=0,
        target_layers=target_layers,
        prefill_trigger_tokens=[-1],
        generate_trigger_tokens=[-1],
    )
    baseline_output = llm.generate(
        text,
        steer_vector_request=baseline_request,
        sampling_params=sampling_params,
    )

    happy_request = SteerVectorRequest(
        "happy",
        2,
        steer_vector_local_path="/app/easysteer/vectors/happy_diffmean.gguf",
        scale=2.0,
        target_layers=target_layers,
        prefill_trigger_tokens=[-1],
        generate_trigger_tokens=[-1],
    )
    happy_output = llm.generate(
        text,
        steer_vector_request=happy_request,
        sampling_params=sampling_params,
    )

    print(baseline_output[0].outputs[0].text)
    print(happy_output[0].outputs[0].text)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
