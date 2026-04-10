import os
from pathlib import Path

from setuptools import setup


def _ensure_cuda_home():
    if os.environ.get("CUDA_HOME"):
        return

    candidates = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path))

    default_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if default_root.exists():
        candidates.extend(sorted(default_root.iterdir(), reverse=True))

    for candidate in candidates:
        if (candidate / "bin" / "nvcc.exe").exists():
            resolved = str(candidate)
            os.environ.setdefault("CUDA_HOME", resolved)
            os.environ.setdefault("CUDA_PATH", resolved)
            break


_ensure_cuda_home()

import torch.utils.cpp_extension as cpp_extension

if cpp_extension.CUDA_HOME is None:
    cpp_extension.CUDA_HOME = cpp_extension._find_cuda_home()

BuildExtension = cpp_extension.BuildExtension
CUDAExtension = cpp_extension.CUDAExtension


MODEL_VARIANTS = {
    "0.8b": {
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "vocab_size": 248320,
        "num_layers": 24,
    },
    "2b": {
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "vocab_size": 248320,
        "num_layers": 24,
    },
}

variant_name = os.environ.get("QWEN_KERNEL_VARIANT", "0.8b").lower()
if variant_name not in MODEL_VARIANTS:
    valid = ", ".join(sorted(MODEL_VARIANTS))
    raise ValueError(f"Unsupported QWEN_KERNEL_VARIANT={variant_name!r}; expected one of: {valid}")
variant = MODEL_VARIANTS[variant_name]

nvcc_defs = [
    f"-DQWEN_HIDDEN_SIZE={variant['hidden_size']}",
    f"-DQWEN_INTERMEDIATE_SIZE={variant['intermediate_size']}",
    f"-DQWEN_VOCAB_SIZE={variant['vocab_size']}",
    f"-DQWEN_NUM_LAYERS={variant['num_layers']}",
]

setup(
    name="qwen35_megakernel_bf16",
    ext_modules=[
        CUDAExtension(
            name="qwen35_megakernel_bf16_C",
            sources=[
                "torch_bindings.cpp",
                "kernel.cu",
                "prefill.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-arch=sm_89",
                    "--use_fast_math",
                    "-std=c++17",
                    *nvcc_defs,
                    "-DNUM_BLOCKS=48",
                    "-DBLOCK_SIZE=512",
                    "-DLM_NUM_BLOCKS=512",
                    "-DLM_BLOCK_SIZE=256",
                ],
            },
            libraries=["cublas"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
