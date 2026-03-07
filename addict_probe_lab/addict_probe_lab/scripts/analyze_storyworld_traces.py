#!/usr/bin/env python
"""Analyze storyworld traces for mechinterp-lite signals and motifs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.mechinterp.clustering import kmeans_numpy
from addict_probe.mechinterp.route_features import feature_correlation_with_outcome
from addict_probe.mechinterp.summaries import summarize_trajectories
from addict_probe.trace_schema import DEFAULT_LAYOUT, read_trace_jsonl, row_to_feature


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze storyworld traces")
    p.add_argument("--trace", required=True, help="Trace JSONL path")
    p.add_argument("--clusters", type=int, default=6, help="Number of kmeans clusters")
    p.add_argument("--out-json", default=None, help="Optional JSON summary output path")
    p.add_argument(
        "--out-clusters-npz",
        default=None,
        help="Optional NPZ output for cluster labels and centroids",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_trace_jsonl(args.trace)
    x = (
        np.stack([row_to_feature(r, layout=DEFAULT_LAYOUT) for r in rows], axis=0)
        if rows
        else np.zeros((0, DEFAULT_LAYOUT.total_dim), dtype=np.float32)
    )

    cluster_out = kmeans_numpy(x, k=args.clusters) if len(x) else {"centroids": x, "labels": np.zeros((0,), dtype=np.int32)}
    summary = summarize_trajectories(args.trace)
    corr_moral = feature_correlation_with_outcome(args.trace, outcome_key="moral_score_proxy")
    corr_useful = feature_correlation_with_outcome(args.trace, outcome_key="usefulness_score_proxy")

    report = {
        "summary": summary,
        "top_feature_corr_moral": corr_moral,
        "top_feature_corr_usefulness": corr_useful,
        "cluster_count": int(cluster_out["centroids"].shape[0]) if len(x) else 0,
        "trace_rows": int(len(rows)),
    }

    text = json.dumps(report, indent=2)
    print(text)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")

    if args.out_clusters_npz:
        Path(args.out_clusters_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.out_clusters_npz,
            labels=cluster_out["labels"],
            centroids=cluster_out["centroids"],
        )


if __name__ == "__main__":
    main()
