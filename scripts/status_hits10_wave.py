#!/usr/bin/env python3
"""Summarize Hits@1/Hits@10 progress for a controlled experiment wave."""

import argparse
import glob
import json
import os
import re
from pathlib import Path


METRIC_RE = re.compile(
    r"Test: epoch=(\d+).*?Hits@1=([0-9.]+).*?Hits@10=([0-9.]+)"
)
PSEUDO_RE = re.compile(
    r"Pseudo pairs: (\d+) \+ (\d+) unique_targets=(\d+)\+(\d+) "
    r"collisions=(\d+)\+(\d+)"
)


def read_metrics(path):
    metrics = []
    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            match = METRIC_RE.search(line)
            if match:
                metrics.append(
                    (int(match.group(1)), float(match.group(2)), float(match.group(3)))
                )
    return metrics


def read_pseudo_stats(path):
    stats = []
    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            match = PSEUDO_RE.search(line)
            if match:
                pair_1, pair_2, unique_1, unique_2, collision_1, collision_2 = (
                    int(value) for value in match.groups()
                )
                total_pairs = pair_1 + pair_2
                total_collisions = collision_1 + collision_2
                stats.append({
                    "pairs": [pair_1, pair_2],
                    "unique_targets": [unique_1, unique_2],
                    "collisions": [collision_1, collision_2],
                    "collision_rate": (
                        float(total_collisions) / total_pairs if total_pairs else 0.0
                    ),
                })
    return stats


def metric_dict(item):
    if item is None:
        return None
    return {"epoch": item[0], "hits1": item[1], "hits10": item[2]}


def metric_at_epoch(metrics, epoch):
    return next((item for item in metrics if item[0] == epoch), None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--root", default="out/zh-hits10")
    parser.add_argument("--hits1-floor", type=float, default=0.8887619047619048)
    parser.add_argument("--baseline-result", type=Path)
    parser.add_argument(
        "--report-epoch",
        type=int,
        help="Require and report this exact epoch for every matched run.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = {}
    if args.baseline_result:
        payload = json.loads(args.baseline_result.read_text())
        baseline = {
            int(item["epoch"]): item for item in payload.get("test_history", ())
        }

    pattern = os.path.join(args.root, args.tag, "zh_en*.log")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit("no run logs matched: {}".format(pattern))

    reports = []
    missing_requested_epoch = []
    for path in paths:
        metrics = read_metrics(path)
        if not metrics:
            print("{} metrics=0".format(os.path.basename(path)))
            continue
        latest = metrics[-1]
        max_h1 = max(metrics, key=lambda item: (item[1], item[2], -item[0]))
        max_h10 = max(metrics, key=lambda item: (item[2], item[1], -item[0]))
        eligible = [item for item in metrics if item[1] >= args.hits1_floor]
        joint = max(eligible, key=lambda item: (item[2], item[1], -item[0])) if eligible else None
        baseline_item = baseline.get(latest[0])
        delta = None
        if baseline_item is not None:
            delta = {
                "hits1": latest[1] - float(baseline_item["hits1"]),
                "hits10": latest[2] - float(baseline_item["hits10"]),
            }
        pseudo = read_pseudo_stats(path)
        requested = (
            metric_at_epoch(metrics, args.report_epoch)
            if args.report_epoch is not None else None
        )
        requested_baseline = (
            baseline.get(args.report_epoch)
            if args.report_epoch is not None else None
        )
        requested_delta = None
        if args.report_epoch is not None and requested is None:
            missing_requested_epoch.append(os.path.basename(path))
        elif requested is not None and requested_baseline is not None:
            requested_delta = {
                "hits1": requested[1] - float(requested_baseline["hits1"]),
                "hits10": requested[2] - float(requested_baseline["hits10"]),
            }
        report = {
            "run_name": os.path.basename(path)[:-4],
            "metric_count": len(metrics),
            "latest": metric_dict(latest),
            "max_hits1": metric_dict(max_h1),
            "max_hits10": metric_dict(max_h10),
            "best_joint": metric_dict(joint),
            "baseline_at_latest": baseline_item,
            "delta_vs_baseline": delta,
            "latest_pseudo": pseudo[-1] if pseudo else None,
            "metric_at_requested_epoch": metric_dict(requested),
            "baseline_at_requested_epoch": requested_baseline,
            "delta_at_requested_epoch": requested_delta,
        }
        reports.append(report)
        print(
            "{} metrics={} latest={} max_h1={} max_h10={} joint={} delta={} pseudo={} "
            "requested={} requested_delta={}".format(
                os.path.basename(path), len(metrics), latest, max_h1, max_h10, joint,
                delta, report["latest_pseudo"], requested, requested_delta,
            )
        )

    if missing_requested_epoch:
        raise SystemExit(
            "requested epoch {} missing from: {}".format(
                args.report_epoch, ", ".join(missing_requested_epoch)
            )
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "tag": args.tag,
            "root": args.root,
            "hits1_floor": args.hits1_floor,
            "report_epoch": args.report_epoch,
            "baseline_result": (
                str(args.baseline_result.resolve()) if args.baseline_result else None
            ),
            "runs": reports,
        }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
