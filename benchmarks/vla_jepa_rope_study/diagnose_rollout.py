#!/usr/bin/env python3
"""Paired post-hoc rollout interventions on the original 18 frozen checkpoints.

No retraining, checkpoint selection, downloads, or repository edits. Input
archives are read as data; checkpoints use torch.load(weights_only=True).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import platform
import statistics
import time
import zipfile
from pathlib import Path

import torch


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def import_experiment(path: Path):
    spec = importlib.util.spec_from_file_location("diagnostic_original_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode_positions(positions: torch.Tensor) -> torch.Tensor:
    """Known observation map phi(p), matching the original pointwise generator."""
    centers = positions.new_tensor([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    point = positions[..., None, :].expand(*positions.shape[:-1], 4, 2)
    local = point - centers
    return torch.cat(
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


def read_position(features: torch.Tensor) -> torch.Tensor:
    return features[..., :2].mean(dim=-2)


def predict_last(model, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return model(frames.flatten(1, 2), actions).reshape(len(frames), -1, 4, 16)[:, -1]


def mse_by_episode(value: torch.Tensor) -> torch.Tensor:
    return value.double().flatten(1).square().mean(dim=1)


def physics_reconstruction(
    errors: list[torch.Tensor], defects: list[torch.Tensor], roundoff: list[torch.Tensor]
) -> dict:
    """Exact physical-coordinate recurrence, retaining generator roundoff."""
    actual = torch.stack(errors, dim=1)
    raw = torch.stack(defects, dim=1)
    reference = torch.stack(roundoff, dim=1)
    driving = raw - reference
    reconstructed = []
    quadratic = []
    for horizon in range(len(errors)):
        weights = driving.new_tensor([(1.0 - 0.85 ** (horizon - j + 1)) / 0.15 for j in range(horizon + 1)])
        weighted = driving[:, : horizon + 1] * weights[None, :, None]
        reconstructed.append(weighted.sum(dim=1))
        diagonal = weighted.square().sum(dim=1).mean()
        offdiagonal = driving.new_zeros(())
        for i in range(horizon + 1):
            for j in range(i + 1, horizon + 1):
                offdiagonal += 2.0 * (weighted[:, i] * weighted[:, j]).mean()
        total = weighted.sum(dim=1).square().mean()
        quadratic_residual = (total - diagonal - offdiagonal).abs()
        if float(quadratic_residual) > 1e-10:
            raise RuntimeError("Physics-defect quadratic decomposition failed numerical identity.")
        quadratic.append(
            {
                "horizon": horizon + 1,
                "effective_defect_count": horizon + 1,
                "temporal_pair_count": horizon * (horizon + 1) // 2,
                "episode_count": len(actual),
                "coordinate_count": actual.shape[-1],
                "physics_kernel_diagonal_mse": float(diagonal),
                "physics_kernel_offdiagonal_mse": float(offdiagonal),
                "physics_kernel_total_mse": float(total),
                "physics_kernel_identity_residual": float(quadratic_residual),
            }
        )
    reconstructed = torch.stack(reconstructed, dim=1)
    residual = (reconstructed - actual).abs().max()
    if float(residual) > 1e-9:
        raise RuntimeError("Physics-defect convolution did not reconstruct coordinate error.")
    lags = {}
    for lag in (1, 2, 4, 8):
        left, right = driving[:, :-lag], driving[:, lag:]
        norm = left.square().sum().sqrt() * right.square().sum().sqrt()
        lags[str(lag)] = float((left * right).sum() / norm) if float(norm) > 0 else None
    return {
        "companion_eigenvalues": [1.0, 0.85],
        "kernel": [(1.0 - 0.85 ** (k + 1)) / 0.15 for k in range(len(errors))],
        "max_absolute_reconstruction_residual": float(residual),
        "reference_roundoff_defect_max_abs": float(reference.abs().max()),
        "driving_defect_uncentered_temporal_alignment_by_lag": lags,
        "quadratic_decomposition_by_horizon": quadratic,
        "quadratic_note": "Diagonal is the weighted energy of individual effective defects. Signed offdiagonal is the explicit sum of all temporal pair cross products; it is not an MSE or a lower-is-better score. Their sum equals physical-coordinate MSE.",
        "position_error_vectors_by_episode_horizon": actual.tolist(),
        "raw_defect_vectors_by_episode_horizon": raw.tolist(),
        "effective_driving_defect_vectors_by_episode_horizon": driving.tolist(),
        "reference_roundoff_vectors_by_episode_horizon": reference.tolist(),
        "reconstructed_position_error_vectors_by_episode_horizon": reconstructed.tolist(),
        "note": "Uncentered temporal alignment includes persistent bias; not an independent-sample correlation test. Eigenvalue 1 permits accumulated drift, not exponential instability.",
    }


@torch.no_grad()
def interventions(model, data: dict, ids: list[int], window: int | None) -> dict:
    z, a, positions = (data[key][ids] for key in ("features", "actions", "positions"))
    length = a.shape[1]
    true_outputs = []
    for t in range(1, length):
        start = 0 if window is None else max(0, t + 1 - window)
        true_outputs.append(predict_last(model, z[:, start : t + 1], a[:, start : t + 1]))
    report = {}
    for feedback in ("teacher", "raw", "projected"):
        context = z[:, :2].clone()
        records = []
        position_errors, defects, roundoffs = [], [], []
        for horizon, t in enumerate(range(1, length), 1):
            start = 0 if window is None else max(0, t + 1 - window)
            proposal = (
                true_outputs[horizon - 1]
                if feedback == "teacher"
                else predict_last(model, context[:, start : t + 1], a[:, start : t + 1])
            )
            position = read_position(proposal)
            reencoded = encode_positions(position)
            fed = reencoded if feedback == "projected" else proposal
            truth = z[:, t + 1]
            active_context = z if feedback == "teacher" else context
            current_p = read_position(active_context[:, t]).double()
            previous_p = read_position(active_context[:, t - 1]).double()
            force = data["raw_actions"][ids, t].double()
            p_error = position.double() - positions[:, t + 1].double()
            defect = position.double() - (1.85 * current_p - 0.85 * previous_p + 0.15 * force)
            eta = positions[:, t + 1].double() - (
                1.85 * positions[:, t].double() - 0.85 * positions[:, t - 1].double() + 0.15 * force
            )
            e_current = current_p - positions[:, t].double()
            e_previous = previous_p - positions[:, t - 1].double()
            recurrence_residual = p_error - (1.85 * e_current - 0.85 * e_previous + defect - eta)
            position_errors.append(p_error)
            defects.append(defect)
            roundoffs.append(eta)
            # Exact algebra in float64 on already computed float32 outputs.
            b = true_outputs[horizon - 1].double() - truth.double()
            d = proposal.double() - true_outputs[horizon - 1].double()
            e = proposal.double() - truth.double()
            bias_mse, shift_mse, total_mse = (mse_by_episode(x) for x in (b, d, e))
            cross = 2.0 * (b * d).flatten(1).mean(dim=1)
            residual = total_mse - bias_mse - shift_mse - cross
            normalized_residual = residual.abs() / (
                total_mse.abs() + bias_mse.abs() + shift_mse.abs() + cross.abs() + 1e-12
            )
            if float(normalized_residual.max()) > 1e-10:
                raise RuntimeError("Exact bias/feedback decomposition failed numerical identity.")
            record = {
                "horizon": horizon,
                "prediction_time_index": t + 1,
                "context_frames": t + 1 - start,
                "context_start_time_index": start,
                "proposal_feature_mse": float(total_mse.mean()),
                "feedback_feature_mse": float(mse_by_episode(fed - truth).mean()),
                "physical_position_mse": float(mse_by_episode(p_error).mean()),
                "physics_raw_defect_mse": float(mse_by_episode(defect).mean()),
                "physics_effective_defect_mse": float(mse_by_episode(defect - eta).mean()),
                "physics_defect_mean_x": float(defect[:, 0].mean()),
                "physics_defect_mean_y": float(defect[:, 1].mean()),
                "physics_recurrence_max_absolute_residual": float(recurrence_residual.abs().max()),
                "known_feature_map_residual_mse": float(mse_by_episode(proposal - reencoded).mean()),
                "patch_position_disagreement_mse": float(
                    mse_by_episode(proposal[..., :2] - position[:, None]).mean()
                ),
                "output_feature_rms": float(proposal.double().square().mean().sqrt()),
                "true_feature_rms": float(truth.double().square().mean().sqrt()),
                "bias_mse": float(bias_mse.mean()),
                "feedback_shift_mse": float(shift_mse.mean()),
                "bias_feedback_cross_term": float(cross.mean()),
                "decomposition_max_normalized_residual": float(normalized_residual.max()),
                "proposal_feature_mse_per_episode": total_mse.tolist(),
                "physical_position_mse_per_episode": mse_by_episode(p_error).tolist(),
            }
            if not all(math.isfinite(v) for v in record.values() if isinstance(v, float)):
                raise RuntimeError("Non-finite diagnostic metric.")
            records.append(record)
            if feedback != "teacher":
                context = torch.cat((context, fed[:, None]), dim=1)
        fields = [key for key, value in records[0].items() if isinstance(value, float)]
        report[feedback] = {
            "horizons": records,
            "mean_15": {key: statistics.mean(r[key] for r in records) for key in fields},
            "mean_7": {key: statistics.mean(r[key] for r in records[:7]) for key in fields},
        }
        if feedback != "teacher":
            report[feedback]["physics_reconstruction"] = physics_reconstruction(
                position_errors, defects, roundoffs
            )
            components = report[feedback]["physics_reconstruction"]["quadratic_decomposition_by_horizon"]
            component_fields = [key for key in components[0] if key.startswith("physics_kernel_")]
            for record, component in zip(records, components, strict=True):
                record.update({key: component[key] for key in component_fields})
            for span, selected in (("mean_7", components[:7]), ("mean_15", components)):
                report[feedback][span].update(
                    {key: statistics.mean(row[key] for row in selected) for key in component_fields}
                )
    return report


@torch.no_grad()
def directional_sensitivity(model, data: dict, ids: list[int]) -> list[dict]:
    """Central finite differences in four fixed directions; not a Jacobian norm."""
    chosen = ids[:4]
    rows = []
    for length in (4, 8, 16):
        frames = data["features"][chosen, :length]
        actions = data["actions"][chosen, :length]
        generator = torch.Generator().manual_seed(19891 + length)
        for kind in ("last_frame", "whole_history"):
            for direction_index in range(4):
                direction = torch.randn(frames.shape, generator=generator)
                if kind == "last_frame":
                    direction[:, :-1] = 0
                direction = direction / direction.flatten(1).norm(dim=1)[:, None, None, None]
                for epsilon in (1e-3, 1e-4):
                    difference = (
                        predict_last(model, frames + epsilon * direction, actions)
                        - predict_last(model, frames - epsilon * direction, actions)
                    ) / (2 * epsilon)
                    gains = difference.flatten(1).norm(dim=1)
                    rows.append(
                        {
                            "context_frames": length,
                            "scope": kind,
                            "direction": direction_index,
                            "epsilon": epsilon,
                            "mean_directional_l2_gain": float(gains.mean()),
                            "max_episode_directional_l2_gain": float(gains.max()),
                        }
                    )
    return rows


def state_envelope(data: dict, split: dict) -> dict:
    training_positions = data["positions"][split["train"], :9].reshape(-1, 2)
    lower, upper = training_positions.amin(0), training_positions.amax(0)
    test = data["positions"][split["test"]]
    outside = ((test < lower) | (test > upper)).any(-1)
    return {
        "training_position_min": lower.tolist(),
        "training_position_max": upper.tolist(),
        "test_fraction_outside_finite_training_rectangle_by_time": outside.float().mean(0).tolist(),
        "note": "Finite sampled axis-aligned envelope, not the mathematical distribution support.",
    }


def oracle_dynamics(data: dict) -> dict:
    position = data["positions"].double()
    history = [position[:, 0], position[:, 1]]
    errors = []
    for t in range(1, 16):
        prediction = 1.85 * history[-1] - 0.85 * history[-2] + 0.15 * data["raw_actions"][:, t].double()
        errors.append(prediction - position[:, t + 1])
        history.append(prediction)
    errors = torch.stack(errors, dim=1)
    return {
        "equation": "p_next = 1.85*p_current - 0.85*p_previous + 0.15*action_current",
        "companion_eigenvalues": [1.0, 0.85],
        "max_absolute_15_step_error": float(errors.abs().max()),
        "physical_mse_by_horizon": errors.square().mean(dim=(0, 2)).tolist(),
        "note": "True two-frame dynamics rolled in float64 against the original float32 generator; residual is generator roundoff.",
    }


def aggregate(runs: list[dict]) -> dict:
    per_seed = []
    modes = [
        (window, feedback) for window in ("full", "8", "2") for feedback in ("teacher", "raw", "projected")
    ]
    for seed in sorted({r["seed"] for r in runs}):
        metrics = {}
        for window, feedback in modes:
            key = f"{window}/{feedback}"
            metrics[key] = {}
            for span in ("mean_7", "mean_15"):
                metrics[key][span] = {}
                fields = runs[0]["interventions"][window][feedback][span]
                for field in fields:
                    values = {}
                    for variant in ("old", "new"):
                        selected = [r for r in runs if r["seed"] == seed and r["variant"] == variant]
                        count = sum(len(r["test_episode_ids"]) for r in selected)
                        values[variant] = (
                            sum(
                                r["interventions"][window][feedback][span][field] * len(r["test_episode_ids"])
                                for r in selected
                            )
                            / count
                        )
                    old, new = values["old"], values["new"]
                    metrics[key][span][field] = {
                        **values,
                        "new_minus_old": new - old,
                    }
                    if field.endswith("_mse") and "offdiagonal" not in field:
                        metrics[key][span][field]["relative_change_percent"] = (
                            (new / old - 1) * 100 if old else None
                        )
        per_seed.append({"seed": seed, "metrics": metrics})
    overall = {}
    for window, feedback in modes:
        key = f"{window}/{feedback}"
        overall[key] = {}
        for span in ("mean_7", "mean_15"):
            overall[key][span] = {}
            for field in per_seed[0]["metrics"][key][span]:
                values = {
                    v: statistics.mean(s["metrics"][key][span][field][v] for s in per_seed)
                    for v in ("old", "new")
                }
                old, new = values["old"], values["new"]
                overall[key][span][field] = {
                    **values,
                    "new_minus_old": new - old,
                    "seeds_with_smaller_numeric_value_new": sum(
                        s["metrics"][key][span][field]["new_minus_old"] < 0 for s in per_seed
                    ),
                }
                if field.endswith("_mse") and "offdiagonal" not in field:
                    overall[key][span][field]["relative_change_percent"] = (
                        (new / old - 1) * 100 if old else None
                    )
    return {
        "per_seed": per_seed,
        "overall": overall,
        "aggregation_note": "Trajectory-weighted folds within seed, descriptive mean across seeds. No significance claim. Signed cross terms and coordinate means are arithmetic quantities, not lower-is-better scores; percentages are restricted to nonnegative MSE fields.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True, help="original results.zip")
    parser.add_argument("--checkpoints", type=Path, required=True, help="original checkpoints.zip")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit-runs", type=int, default=0, help="smoke only; 0 means all 18")
    args = parser.parse_args()
    if args.limit_runs < 0 or args.limit_runs > 18 or args.limit_runs % 2:
        parser.error("--limit-runs must be 0 or an even number up to 18, preserving complete old/new pairs.")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "summary.json").exists():
        raise RuntimeError("Use a fresh diagnostic output directory; existing results are not overwritten.")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    experiment_path = Path(__file__).with_name("experiment.py")
    experiment = import_experiment(experiment_path)
    modules, source = experiment.load_modules(args.repo)
    data = experiment.generate_data(120, 16, 62026)
    if not torch.equal(encode_positions(data["positions"]), data["features"]):
        raise RuntimeError("Diagnostic feature map differs from original generator.")
    started = time.perf_counter()
    runs = []
    with zipfile.ZipFile(args.results) as result_zip, zipfile.ZipFile(args.checkpoints) as checkpoint_zip:
        manifest = json.loads(result_zip.read("results/full/manifest.json"))
        regenerated_data_hash = experiment.state_hash(data)
        original_data_hash = manifest["identity"]["data_tensor_sha256"]
        # Cross-platform CPU transcendental kernels need not be bit-identical.
        # Preserve both hashes and require every saved CUDA rollout to reproduce
        # numerically below; never pretend the regenerated data are identical.
        expected_config = {
            "episodes": 120,
            "trajectory_length": 16,
            "data_seed": 62026,
            "training_sequence_length": 8,
        }
        if any(manifest["identity"]["config"].get(key) != value for key, value in expected_config.items()):
            raise RuntimeError("Original data-generating configuration differs.")
        if sha(experiment_path.read_bytes()) != manifest["identity"]["config"]["script_sha256"]:
            raise RuntimeError("Local experiment script differs from the trained script.")
        paths = sorted(
            n
            for n in result_zip.namelist()
            if n.startswith("results/full/fold_") and n.endswith("/result.json")
        )
        if len(paths) != 18:
            raise RuntimeError("Expected exactly 18 formal original runs.")
        if args.limit_runs:
            paths = paths[: args.limit_runs]
        for path in paths:
            result = json.loads(result_zip.read(path))
            checkpoint_bytes = checkpoint_zip.read(path.replace("result.json", "best.pt"))
            if sha(checkpoint_bytes) != result["checkpoint_sha256"]:
                raise RuntimeError("Checkpoint hash mismatch.")
            payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
            model = experiment.new_model(modules[result["variant"]]).eval()
            model.load_state_dict(payload["model"], strict=True)
            interventions_by_window = {}
            for window in (None, 8, 2):
                interventions_by_window["full" if window is None else str(window)] = interventions(
                    model, data, result["test_episode_ids"], window
                )
            original = result["autoregressive_mse_by_horizon"]
            reproduced = [
                r["proposal_feature_mse"] for r in interventions_by_window["full"]["raw"]["horizons"]
            ]
            max_error = max(abs(a - b) for a, b in zip(original, reproduced, strict=True))
            if max_error > 1e-5:
                raise RuntimeError(
                    f"CPU reproduction mismatch {max_error}; do not reinterpret a different rollout."
                )
            run = {
                "fold": result["fold"],
                "seed": result["seed"],
                "variant": result["variant"],
                "test_episode_ids": result["test_episode_ids"],
                "checkpoint_sha256": sha(checkpoint_bytes),
                "original_cuda_ar_max_absolute_reproduction_error": max_error,
                "interventions": interventions_by_window,
                "directional_sensitivity": directional_sensitivity(model, data, result["test_episode_ids"]),
                "state_envelope": state_envelope(data, result["split"]),
            }
            runs.append(run)
            save_json(args.out / f"fold_{run['fold']}_seed_{run['seed']}_{run['variant']}.json", run)
            print(
                f"completed {len(runs)}/{len(paths)} fold={run['fold']} seed={run['seed']} variant={run['variant']} reproduction_error={max_error:.3g}",
                flush=True,
            )
    summary = aggregate(runs)
    summary.update(
        {
            "complete_18_runs": len(runs) == 18,
            "run_count": len(runs),
            "elapsed_seconds": time.perf_counter() - started,
            "provenance": {
                "diagnostic_script_sha256": sha(Path(__file__).read_bytes()),
                "original_experiment_sha256": sha(experiment_path.read_bytes()),
                "data_tensor_sha256": experiment.state_hash(data),
                "original_windows_data_tensor_sha256": original_data_hash,
                "data_bit_identical_to_original": regenerated_data_hash == original_data_hash,
                "data_regeneration_note": "Same audited generator/configuration; numerical regeneration on Mac CPU. Hash can differ across CPU implementations. All saved original CUDA rollout horizons must reproduce within 1e-5 absolute MSE.",
                "checkpoint_archive_sha256": sha(args.checkpoints.read_bytes()),
                "results_archive_sha256": sha(args.results.read_bytes()),
                "source": source,
                "torch": str(torch.__version__),
                "python": platform.python_version(),
                "device": "cpu",
                "threads": args.threads,
            },
            "limitations": [
                "Post-hoc diagnostics on all frozen original checkpoints; no retraining or result replacement.",
                "Rolling windows both discard history and reset RoPE position indices; not a pure length intervention.",
                "Projection is phi(mean_patch(first_two_coordinates)), not nearest/orthogonal projection; includes averaging.",
                "Feature residual reductions do not uniquely prove off-manifold feedback caused the original difference.",
                "Finite directional gains cover chosen directions/examples, not a full Jacobian spectral norm or stability proof.",
                "Late teacher-forcing jointly changes context length, absolute position indices and state distributions.",
                "Exact bias/shift/cross decomposition is algebraic, not identification of a unique root cause.",
                "Raw feature perturbation gains depend on feature coordinate scaling; values above one do not establish instability. True position dynamics has eigenvalues 1 and 0.85.",
            ],
            "known_dynamics_oracle": oracle_dynamics(data),
        }
    )
    save_json(args.out / "summary.json", summary)
    with (args.out / "horizons.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = []
        for run in runs:
            for window, modes in run["interventions"].items():
                for feedback, details in modes.items():
                    for row in details["horizons"]:
                        rows.append(
                            {
                                "fold": run["fold"],
                                "seed": run["seed"],
                                "variant": run["variant"],
                                "window": window,
                                "feedback": feedback,
                                **{key: value for key, value in row.items() if not isinstance(value, list)},
                            }
                        )
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {"complete_18_runs": summary["complete_18_runs"], "elapsed_seconds": summary["elapsed_seconds"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
