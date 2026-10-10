#!/usr/bin/env python3
"""Verify transferred training evidence without executing archived scripts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath

import torch


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify(root: Path) -> dict:
    checksums = json.loads((root / "checksums.json").read_text())
    manifest = json.loads((root / "archive-manifest.json").read_text())
    assert sha((root / "checksums.json").read_bytes()) == manifest["checksums_sha256"]
    payloads = 0
    formal = []
    for archive_name in ("results.zip", "checkpoints.zip"):
        raw = (root / archive_name).read_bytes()
        expected = manifest["archives"][archive_name]
        assert sha(raw) == expected["sha256"]
        assert len(raw) == expected["size_bytes"]
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            assert archive.testzip() is None
            names = archive.namelist()
            assert len(names) == len(set(names)) == len({n.lower() for n in names})
            assert len(names) == expected["member_count"]
            for info in archive.infolist():
                path = PurePosixPath(info.filename)
                assert not path.is_absolute() and ".." not in path.parts and "\\" not in info.filename
                assert not stat.S_ISLNK(info.external_attr >> 16)
                content = archive.read(info)
                if info.filename == "checksums.json":
                    assert content == (root / "checksums.json").read_bytes()
                    continue
                item = checksums["archives"][archive_name][info.filename]
                assert sha(content) == item["sha256"] and len(content) == item["size_bytes"]
                payloads += 1
                if archive_name == "checkpoints.zip" and info.filename.endswith("/best.pt"):
                    checkpoint = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
                    assert all(bool(torch.isfinite(t).all()) for t in checkpoint["model"].values())
                    assert sum(t.numel() for t in checkpoint["model"].values()) == 153808
                    if info.filename.startswith("results/full/"):
                        formal.append(checkpoint["metadata"])
            if archive_name == "results.zip":
                for local, archived in (
                    ("training-report.txt", "results/full/report.txt"),
                    ("training-summary.json", "results/full/summary.json"),
                    ("runs-index.json", "results/full/runs-index.json"),
                ):
                    if (root / local).exists():
                        assert (root / local).read_bytes() == archive.read(archived)
    assert len(formal) == 18 and payloads == 125
    pairs = {}
    for item in formal:
        key = (item["fold"], item["seed"])
        pairs.setdefault(key, {})[item["variant"]] = item
    assert len(pairs) == 9
    for pair in pairs.values():
        assert set(pair) == {"old", "new"}
        old, new = pair["old"], pair["new"]
        for key in ("initial_state_sha256", "batch_order_sha256", "split"):
            assert old[key] == new[key]
        assert old["best_step"] == new["best_step"] == 1000
    diagnostic_rows = None
    if (root / "diagnostics.zip").exists():
        with zipfile.ZipFile(root / "diagnostics.zip") as archive:
            assert archive.testzip() is None
            summary = json.loads(archive.read("diagnostics/summary.json"))
            assert (
                archive.read("diagnostics/summary.json") == (root / "diagnostics-summary.json").read_bytes()
            )
            assert summary["run_count"] == 18 and summary["complete_18_runs"]
            assert summary["provenance"]["diagnostic_script_sha256"] == sha(
                (root / "diagnose_rollout.py").read_bytes()
            )
            assert summary["provenance"]["original_experiment_sha256"] == sha(
                (root / "experiment.py").read_bytes()
            )
            rows = list(csv.DictReader(io.StringIO(archive.read("diagnostics/horizons.csv").decode())))
            keys = [
                (r["fold"], r["seed"], r["variant"], r["window"], r["feedback"], r["horizon"]) for r in rows
            ]
            assert len(keys) == len(set(keys)) == 2430
            diagnostic_rows = len(keys)
    if (root / "publication-checksums.json").exists():
        package = json.loads((root / "publication-checksums.json").read_text())
        for name, expected in package["files"].items():
            data = (root / name).read_bytes()
            assert sha(data) == expected["sha256"] and len(data) == expected["size_bytes"]
    return {
        "archives_verified": 2,
        "payloads_verified": payloads,
        "formal_checkpoints_verified": len(formal),
        "paired_initializations_batches_splits_verified": len(pairs),
        "finite_checkpoint_tensors": True,
        "no_archived_code_executed": True,
        "diagnostic_rows_verified": diagnostic_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify(args.artifacts.resolve())
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
