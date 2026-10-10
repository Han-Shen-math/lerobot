#!/usr/bin/env python3
"""Controlled old/new RoPE comparison; no model downloads or repository edits.

This is a synthetic latent-dynamics experiment, not a VLA or robot benchmark.
Requires Python 3.12 and PyTorch only. See --help for execution options.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

PINNED_COMMIT = "b9cb121cb7d4e3e26ec5c906d08dda68d151ba52"
PINNED_SOURCE_SHA256 = "050b6bc0901c59fff8efaa5c3d18a6fe35d8ffe968c91c50624b57763d6f847c"


OLD_LINES = (
    "emb_sin = freqs.sin().squeeze(-1).repeat(1, 1, 1, 2)",
    "emb_cos = freqs.cos().squeeze(-1).repeat(1, 1, 1, 2)",
)
NEW_LINES = (
    "emb_sin = freqs.sin().repeat_interleave(2, dim=-1)",
    "emb_cos = freqs.cos().repeat_interleave(2, dim=-1)",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def load_modules(repo: Path) -> tuple[dict, dict]:
    relative = "src/lerobot/policies/vla_jepa/world_model.py"
    path = repo / relative
    raw = path.read_bytes()
    source = raw.decode("utf-8").replace("\r\n", "\n")
    if git(repo, "rev-parse", "HEAD") != PINNED_COMMIT:
        raise RuntimeError(f"Checkout must be pinned to {PINNED_COMMIT}; do not reuse another PR branch.")
    if digest(source.encode()) != PINNED_SOURCE_SHA256:
        raise RuntimeError("The pinned world_model.py source hash does not match.")
    tracked = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{relative}"])
    if tracked.decode("utf-8").replace("\r\n", "\n") != source:
        raise RuntimeError("world_model.py differs from HEAD; use a clean isolated checkout.")
    tree = ast.parse(source)
    helper = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "rotate_queries_or_keys"), None
    )
    if helper is None:
        raise RuntimeError("The expected original RoPE helper is absent.")
    function_source = ast.get_source_segment(source, helper)
    for line in OLD_LINES:
        if source.count(line) != 1 or function_source.count(line) != 1:
            raise RuntimeError("Unexpected source: the exact original two-line bug must be present.")
    corrected = source
    for old, new in zip(OLD_LINES, NEW_LINES, strict=True):
        corrected = corrected.replace(old, new, 1)
    ast.parse(corrected)
    modules = {}
    for variant, text in (("old", source), ("new", corrected)):
        name = f"controlled_rope_{variant}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        # Compile each isolated namespace: never alter the repository source.
        exec(compile(text, str(path) + f"::{variant}", "exec"), module.__dict__)
        modules[variant] = module
    return modules, {
        "git_commit": git(repo, "rev-parse", "HEAD"),
        "source_path": relative,
        "source_sha256_raw": digest(raw),
        "source_sha256_normalized": digest(source.encode()),
        "corrected_source_sha256": digest(corrected.encode()),
        "replacement": [list(pair) for pair in zip(OLD_LINES, NEW_LINES, strict=True)],
        "working_tree_status": git(repo, "status", "--short"),
    }


def reference_rotation(x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    omega = 10000.0 ** (-torch.arange(dim // 2, dtype=x.dtype, device=x.device) / (dim / 2))
    angle = position.to(x.dtype)[..., None] * omega
    pairs = x.unflatten(-1, (-1, 2))
    a, b = pairs.unbind(-1)
    return torch.stack((a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos()), -1).flatten(-2)


def mathematical_checks(modules: dict, device: torch.device) -> dict:
    generator = torch.Generator().manual_seed(9071)
    x = torch.randn(2, 2, 5, 8, generator=generator).to(device)
    k = torch.randn(2, 2, 5, 8, generator=generator).to(device)
    pos = torch.arange(5, device=device, dtype=torch.float32)
    report = {}
    for variant, module in modules.items():
        rotate = module.rotate_queries_or_keys
        tests = {}

        def check(name, operation, tolerance=3e-5):
            try:
                error = float(operation())
                tests[name] = {
                    "passed": math.isfinite(error) and error <= tolerance,
                    "max_error": error,
                    "tolerance": tolerance,
                }
            except Exception as exc:
                tests[name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}

        check("reference_rotation", lambda: (rotate(x, pos) - reference_rotation(x, pos)).abs().max())
        check("norm_preservation", lambda: (rotate(x, pos).norm(dim=-1) - x.norm(dim=-1)).abs().max())
        check(
            "same_position_inner_product",
            lambda: ((rotate(x, pos) * rotate(k, pos)).sum(-1) - (x * k).sum(-1)).abs().max(),
        )
        check(
            "relative_position_joint_shift",
            lambda: (
                (
                    (rotate(x, pos) * rotate(k, pos + 2)).sum(-1)
                    - (rotate(x, pos + 11) * rotate(k, pos + 13)).sum(-1)
                )
                .abs()
                .max()
            ),
            1e-4,
        )
        check(
            "single_frequency_dim2",
            lambda: (rotate(x[..., :2], pos) - reference_rotation(x[..., :2], pos)).abs().max(),
        )

        def gradient_error():
            a = x.double().detach().requires_grad_(True)
            b = a.detach().clone().requires_grad_(True)
            weights = k.double()
            (ga,) = torch.autograd.grad((rotate(a, pos.double()) * weights).sum(), a)
            (gb,) = torch.autograd.grad((reference_rotation(b, pos.double()) * weights).sum(), b)
            return (ga - gb).abs().max()

        check("gradient_reference", gradient_error, 1e-10)

        def attention_finite():
            torch.manual_seed(51)
            attention = module.ACRoPEAttention(dim=96, num_heads=4).to(device)
            z = torch.randn(2, 4 * 5, 96, generator=generator).to(device).requires_grad_(True)
            mask = module.build_action_block_causal_attention_mask(4, 2, 2).to(device)
            out = attention(z, attn_mask=mask, num_frames=4, grid_height=2, grid_width=2, action_tokens=1)
            (grad,) = torch.autograd.grad(out.square().mean(), z)
            return 0.0 if bool(torch.isfinite(out).all() & torch.isfinite(grad).all()) else float("inf")

        check("attention_forward_backward_finite", attention_finite)

        def causal_prefix_error():
            torch.manual_seed(51)
            model = new_model(module).to(device).eval()
            frames = torch.randn(2, 4, 4, 16, generator=generator).to(device)
            actions = torch.randn(2, 4, 8, generator=generator).to(device)
            changed_frames, changed_actions = frames.clone(), actions.clone()
            changed_frames[:, 3] += 7.0
            changed_actions[:, 3] -= 5.0
            with torch.no_grad():
                original = model(frames.flatten(1, 2), actions)[:, :12]
                changed = model(changed_frames.flatten(1, 2), changed_actions)[:, :12]
            return (original - changed).abs().max()

        check("future_input_perturbation_preserves_prefix", causal_prefix_error)
        report[variant] = tests
    if not all(item["passed"] for item in report["new"].values()):
        raise RuntimeError("Corrected implementation failed mathematical checks: " + json.dumps(report))
    return report


def generate_data(episodes: int = 120, length: int = 16, seed: int = 62026) -> dict:
    """CPU-only deterministic trajectories; hidden velocity, visible position.

    v[t+1] = 0.85*v[t] + a[t], p[t+1] = p[t] + 0.15*v[t+1].
    Fixed pointwise features never mix time steps. Targets are z[t+1].
    """
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(episodes, 2, generator=g) * 2 - 1
    v = torch.rand(episodes, 2, generator=g) - 0.5
    actions = (torch.rand(episodes, length, 2, generator=g) * 2 - 1) * 0.35
    positions = [p.clone()]
    for t in range(length):
        v = 0.85 * v + actions[:, t]
        p = p + 0.15 * v
        positions.append(p.clone())
    positions = torch.stack(positions, dim=1)
    centers = torch.tensor([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    point = positions[:, :, None, :].expand(-1, -1, 4, -1)
    local = point - centers
    features = torch.cat(
        (
            point,
            point.square(),
            (math.pi * point).sin(),
            (math.pi * point).cos(),
            local,
            local.sin(),
            local.cos(),
            (-local.square().sum(-1, keepdim=True)).exp(),
            torch.ones_like(local[..., :1]),
        ),
        dim=-1,
    )
    embedded_actions = torch.cat((actions, actions.square(), actions.sin(), actions.cos()), dim=-1)
    return {"features": features, "actions": embedded_actions, "positions": positions, "raw_actions": actions}


def make_splits(episodes: int, folds: int) -> list[dict]:
    if not 2 <= folds <= episodes // 6:
        raise ValueError("Full experiment needs at least 2 folds and sufficient episodes.")
    g = torch.Generator().manual_seed(47291)
    order = torch.randperm(episodes, generator=g)
    tests = torch.tensor_split(order, folds)
    splits = []
    for fold in range(folds):
        train_pool = torch.cat([ids for j, ids in enumerate(tests) if j != fold])
        val_size = max(3, len(train_pool) // 5)
        splits.append(
            {
                "fold": fold,
                "train": train_pool[val_size:].tolist(),
                "validation": train_pool[:val_size].tolist(),
                "test": tests[fold].tolist(),
            }
        )
    return splits


def new_model(module):
    return module.ActionConditionedVideoPredictor(
        num_frames=16,
        img_size=(2, 2),
        patch_size=1,
        tubelet_size=1,
        embed_dim=16,
        action_embed_dim=8,
        predictor_embed_dim=96,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        num_action_tokens_per_step=1,
        dropout=0.0,
    )


def state_hash(state: dict) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(bytes(t.view(torch.uint8).flatten().tolist()))
    return h.hexdigest()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def errors(model, data, ids, horizon=8) -> tuple[float, list[float]]:
    model.eval()
    z = data["features"][ids, : horizon + 1]
    a = data["actions"][ids, :horizon]
    predicted = model(z[:, :-1].flatten(1, 2), a).reshape_as(z[:, 1:])
    per_episode = (predicted - z[:, 1:]).square().mean(dim=(1, 2, 3))
    if not bool(torch.isfinite(per_episode).all()):
        raise RuntimeError("Non-finite evaluation predictions.")
    return float(per_episode.mean()), per_episode.cpu().tolist()


@torch.no_grad()
def evaluate(model, data, ids) -> dict:
    model.eval()
    length = data["actions"].shape[1]
    one_step, episode_mse = errors(model, data, ids, 8)
    long_context, long_episode_mse = errors(model, data, ids, length)
    z = data["features"][ids]
    a = data["actions"][ids]
    persistence = float((z[:, :8] - z[:, 1:9]).square().mean())
    # Explicit temporal mismatch; unchanged observations and targets.
    shuffled_a = torch.roll(a[:, :8], shifts=3, dims=1)
    wrong = model(z[:, :8].flatten(1, 2), shuffled_a).reshape_as(z[:, 1:9])
    shuffled_mse = float((wrong - z[:, 1:9]).square().mean())
    # Two real observations establish velocity; afterwards no true future is fed.
    context = z[:, :2].clone()
    horizon_errors = []
    for t in range(1, length):
        out = model(context.flatten(1, 2), a[:, : t + 1])
        next_z = out.reshape(len(ids), t + 1, 4, 16)[:, -1]
        horizon_errors.append(float((next_z - z[:, t + 1]).square().mean()))
        context = torch.cat((context, next_z[:, None]), dim=1)
    if not all(
        math.isfinite(x) for x in [one_step, long_context, persistence, shuffled_mse, *horizon_errors]
    ):
        raise RuntimeError("Non-finite held-out metrics.")
    return {
        "test_episode_ids": ids,
        "one_step_mse": one_step,
        "one_step_mse_per_episode": episode_mse,
        "one_step_context_frames": 8,
        "long_context_one_step_mse": long_context,
        "long_context_one_step_mse_per_episode": long_episode_mse,
        "long_context_frames": length,
        "long_context_note": "Length extrapolation: trained with 8 context frames, evaluated with 16.",
        "persistence_mse": persistence,
        "temporally_mismatched_actions_mse": shuffled_mse,
        "action_mismatch_mse_increase": shuffled_mse - one_step,
        "autoregressive_prefix_frames": 2,
        "autoregressive_mse_by_horizon": horizon_errors,
        "autoregressive_same_length_mean_mse": sum(horizon_errors[:7]) / 7,
        "autoregressive_mean_mse": sum(horizon_errors) / len(horizon_errors),
    }


def run_training(module, variant, initial, initial_hash, split, seed, data, config, directory):
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists() and config["resume"]:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("complete") and (directory / "best.pt").exists():
            expected = {
                "variant": variant,
                "fold": split["fold"],
                "seed": seed,
                "steps": config["steps"],
                "initial_state_sha256": initial_hash,
                "split": split,
            }
            if any(result.get(key) != value for key, value in expected.items()):
                raise RuntimeError("Completed run metadata differs from the requested paired run.")
            if result.get("checkpoint_sha256") != digest((directory / "best.pt").read_bytes()):
                raise RuntimeError("Completed checkpoint hash does not match; do not silently resume.")
            return result
    torch.manual_seed(seed)
    device = data["features"].device
    model = new_model(module).to(device)
    model.load_state_dict(initial)
    if state_hash(model.state_dict()) != initial_hash:
        raise RuntimeError("Initial state does not match paired run.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed + 100003 * (split["fold"] + 1))
    indices = torch.tensor(split["train"], dtype=torch.long)
    batches = indices[torch.randint(len(indices), (config["steps"], config["batch_size"]), generator=g)]
    batch_hash = digest(bytes(batches.contiguous().view(torch.uint8).flatten().tolist()))
    best_val = float("inf")
    best_state = None
    best_step = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    with (directory / "curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "train_mse", "gradient_norm", "validation_mse"])
        writer.writeheader()
        for step, ids in enumerate(batches, 1):
            model.train()
            ids = ids.to(device)
            z = data["features"][ids, :9]
            a = data["actions"][ids, :8]
            output = model(z[:, :-1].flatten(1, 2), a).reshape_as(z[:, 1:])
            loss = (output - z[:, 1:]).square().mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite loss in {variant}, step {step}.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            if not bool(torch.isfinite(norm)):
                raise RuntimeError(f"Non-finite gradient in {variant}, step {step}.")
            optimizer.step()
            val = None
            if step == 1 or step % 100 == 0 or step == config["steps"]:
                val, _ = errors(model, data, split["validation"], 8)
                if val < best_val:
                    best_val, best_step = val, step
                    best_state = {
                        key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()
                    }
            writer.writerow(
                {
                    "step": step,
                    "train_mse": float(loss.detach()),
                    "gradient_norm": float(norm),
                    "validation_mse": val,
                }
            )
            if val is not None:
                handle.flush()
                print(
                    f"fold={split['fold']} seed={seed} variant={variant} step={step} val={val:.6g}",
                    flush=True,
                )
    synchronize(device)
    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    metrics = evaluate(model, data, split["test"])
    synchronize(device)
    result = {
        "complete": True,
        "variant": variant,
        "fold": split["fold"],
        "seed": seed,
        "steps": config["steps"],
        "best_step": best_step,
        "best_validation_mse": best_val,
        "checkpoint_selection": "minimum validation MSE; test never used for selection",
        "initial_state_sha256": initial_hash,
        "batch_order_sha256": batch_hash,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "all_training_losses_and_gradient_norms_finite": True,
        "training_seconds": training_seconds,
        "total_run_seconds": time.perf_counter() - started,
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None,
        "split": split,
        **metrics,
    }
    torch.save({"model": best_state, "metadata": result}, directory / "best.pt")
    result["checkpoint_sha256"] = digest((directory / "best.pt").read_bytes())
    write_json(result_path, result)  # Completion marker is written last.
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def summarize(results, output, config):
    pairs = []
    for fold, seed in sorted({(r["fold"], r["seed"]) for r in results}):
        matched = {r["variant"]: r for r in results if r["fold"] == fold and r["seed"] == seed}
        if set(matched) != {"old", "new"}:
            continue
        old, new = matched["old"], matched["new"]
        if (
            old["initial_state_sha256"] != new["initial_state_sha256"]
            or old["batch_order_sha256"] != new["batch_order_sha256"]
        ):
            raise RuntimeError("Paired run initialization or batch order differs.")
        pairs.append(
            {
                "fold": fold,
                "seed": seed,
                "test_episode_count": len(old["test_episode_ids"]),
                "one_step_mse_old": old["one_step_mse"],
                "one_step_mse_new": new["one_step_mse"],
                "one_step_mse_new_minus_old": new["one_step_mse"] - old["one_step_mse"],
                "autoregressive_mse_old": old["autoregressive_mean_mse"],
                "autoregressive_mse_new": new["autoregressive_mean_mse"],
                "autoregressive_mse_new_minus_old": new["autoregressive_mean_mse"]
                - old["autoregressive_mean_mse"],
            }
        )
    if len(pairs) != config["folds"] * len(config["seeds"]):
        raise RuntimeError("Incomplete paired runs cannot count as a complete experiment.")
    per_seed = []
    for seed in config["seeds"]:
        seed_pairs = [p for p in pairs if p["seed"] == seed]
        count = sum(p["test_episode_count"] for p in seed_pairs)
        per_seed.append(
            {
                "seed": seed,
                "test_episode_count": count,
                "one_step_delta": sum(
                    p["one_step_mse_new_minus_old"] * p["test_episode_count"] for p in seed_pairs
                )
                / count,
                "autoregressive_delta": sum(
                    p["autoregressive_mse_new_minus_old"] * p["test_episode_count"] for p in seed_pairs
                )
                / count,
            }
        )
    summary = {
        "complete": True,
        "pairs": pairs,
        "completed_runs": len(results),
        "per_seed_trajectory_weighted_fold_aggregation": per_seed,
        "mean_paired_one_step_mse_delta": statistics.mean(p["one_step_delta"] for p in per_seed),
        "std_across_initialization_seeds_one_step_delta": statistics.stdev(
            p["one_step_delta"] for p in per_seed
        )
        if len(per_seed) > 1
        else None,
        "mean_paired_autoregressive_mse_delta": statistics.mean(p["autoregressive_delta"] for p in per_seed),
        "std_across_initialization_seeds_autoregressive_delta": statistics.stdev(
            p["autoregressive_delta"] for p in per_seed
        )
        if len(per_seed) > 1
        else None,
        "aggregation_note": "First weight held-out trajectories across folds within each initialization seed, then describe seed mean/sample std. No independence or significance claim.",
        "interpretation": "Negative deltas favor corrected RoPE on this synthetic task only.",
        "limitations": [
            "Synthetic fixed features, not real video or full VLA.",
            "No pretrained-checkpoint or robot-performance claim.",
            "Folds share training data; seeds and trajectory windows are not independent new datasets.",
            "No significance test or pseudoreplicated confidence interval is reported.",
            "Training dynamics can compensate for the old formula; improvement is not guaranteed.",
        ],
    }
    if config["mode"] == "pilot":
        summary["projected_18_run_training_seconds"] = (
            sum(r["training_seconds"] for r in results) * 9 * (1000 / config["steps"])
        )
        summary["projection_note"] = (
            "Rough estimate for 18 runs x 1000 steps, excludes evaluation/setup and is not a runtime guarantee."
        )
    write_json(output / "summary.json", summary)
    lines = [
        "Controlled RoPE old/new experiment",
        f"Mode: {config['mode']}",
        f"Completed runs: {len(results)}; paired comparisons: {len(pairs)}",
        "Negative new-minus-old error differences favor corrected RoPE.",
    ]
    for pair in pairs:
        lines.append(
            f"fold={pair['fold']} seed={pair['seed']}: one-step delta={pair['one_step_mse_new_minus_old']:.8g}; autoregressive delta={pair['autoregressive_mse_new_minus_old']:.8g}"
        )
    lines.extend(["", "PER-SEED CROSS-FOLD AGGREGATES (trajectory-weighted)"])
    for item in per_seed:
        lines.append(
            f"seed={item['seed']}: one-step delta={item['one_step_delta']:.8g}; autoregressive delta={item['autoregressive_delta']:.8g}"
        )
    lines.append(
        f"Seed mean one-step delta: {summary['mean_paired_one_step_mse_delta']:.8g}; sample std: {summary['std_across_initialization_seeds_one_step_delta']}"
    )
    lines.append(
        "Primary one-step metric uses 8 frames, matching training. Sixteen-frame teacher-forced/AR metrics include context-length extrapolation."
    )
    lines.extend(["", "LIMITATIONS", *summary["limitations"]])
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seeds", default="11,22,33", help="comma-separated paired initialization seeds")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--pilot", action="store_true", help="200 steps, one paired run, in OUT/pilot")
    parser.add_argument("--smoke", action="store_true", help="2 steps, one paired run, in OUT/smoke")
    parser.add_argument(
        "--resume", action="store_true", help="skip completed matching runs; restart incomplete runs"
    )
    args = parser.parse_args()
    if args.pilot and args.smoke:
        parser.error("Choose pilot or smoke, not both.")
    seeds = [int(part) for part in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds) or args.steps < 1:
        parser.error("Seeds must be unique and steps positive.")
    mode = "smoke" if args.smoke else "pilot" if args.pilot else "full"
    device = torch.device(args.device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; there is no silent CPU fallback.")
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    modules, source = load_modules(args.repo.resolve())
    output = args.out.resolve() / mode if mode != "full" else args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    folds = args.folds if mode == "full" else 1
    selected_seeds = seeds if mode == "full" else seeds[:1]
    config = {
        "mode": mode,
        "folds": folds,
        "seeds": selected_seeds,
        "steps": 2 if mode == "smoke" else 200 if mode == "pilot" else args.steps,
        "batch_size": 16,
        "episodes": 120,
        "trajectory_length": 16,
        "training_sequence_length": 8,
        "device": str(device),
        "dtype": "float32",
        "dropout": 0.0,
        "data_seed": 62026,
        "generator_sha256": digest(inspect.getsource(generate_data).encode()),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "resume": args.resume,
    }
    compatible_config = {key: value for key, value in config.items() if key != "resume"}
    cpu_data = generate_data(config["episodes"], config["trajectory_length"], config["data_seed"])
    data_hash = state_hash(cpu_data)
    stable_environment = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "deterministic_algorithms": True,
        "tf32": False,
        "sdpa_backend": "math",
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    if device.type == "cuda":
        stable_environment.update(
            {
                "gpu": torch.cuda.get_device_name(device),
                "compute_capability": list(torch.cuda.get_device_capability(device)),
                "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
        )
    identity = {
        "config": compatible_config,
        "source": source,
        "environment": stable_environment,
        "data_tensor_sha256": data_hash,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume or existing["identity"] != identity:
            raise RuntimeError(
                "Output exists; use --resume only with matching config and source, or a new --out."
            )
    else:
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "cpu_threads": torch.get_num_threads(),
        }
        if device.type == "cuda":
            environment.update(
                {
                    "gpu": torch.cuda.get_device_name(device),
                    "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
                    "compute_capability": list(torch.cuda.get_device_capability(device)),
                }
            )
            try:
                environment["nvidia_smi"] = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                    text=True,
                ).strip()
            except (OSError, subprocess.CalledProcessError) as exc:
                environment["nvidia_smi_error"] = str(exc)
        write_json(manifest_path, {"identity": identity, "environment": environment})
    math_report = mathematical_checks(modules, device)
    write_json(output / "math_checks.json", math_report)
    # Always form three trajectory groups for one-pair smoke/pilot.
    splits = make_splits(config["episodes"], args.folds if mode == "full" else 3)[:folds]
    write_json(output / "splits.json", splits)
    data = {key: value.to(device) for key, value in cpu_data.items()}
    results = []
    for split in splits:
        for seed in selected_seeds:
            torch.manual_seed(seed)
            initial = copy.deepcopy(new_model(modules["old"]).state_dict())
            initial_hash = state_hash(initial)
            for variant in ("old", "new"):
                directory = output / f"fold_{split['fold']}_seed_{seed}_{variant}"
                results.append(
                    run_training(
                        modules[variant], variant, initial, initial_hash, split, seed, data, config, directory
                    )
                )
    summary = summarize(results, output, config)
    print(
        json.dumps(
            {
                "output": str(output),
                "complete": summary["complete"],
                "completed_runs": summary["completed_runs"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
