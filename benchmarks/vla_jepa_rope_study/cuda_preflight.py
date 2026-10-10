"""Record the real CUDA runtime before the paired experiment; never fall back to CPU."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    info: dict[str, object] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        snapshot = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False, timeout=15)
        info["nvidia_smi_exit_code"] = snapshot.returncode
        info["nvidia_smi"] = snapshot.stdout + snapshot.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        info["nvidia_smi_error"] = str(exc)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_vram_bytes"] = props.total_memory
        info["compute_capability"] = [props.major, props.minor]
        sample = torch.tensor([1.0, 2.0], device="cuda")
        actual = (sample * sample).cpu().tolist()
        torch.cuda.synchronize()
        info["cuda_arithmetic"] = actual
        info["passed"] = actual == [1.0, 4.0]
    else:
        info["passed"] = False
        info["error"] = "CUDA unavailable. CPU fallback is not GPU evidence."
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if not info["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
