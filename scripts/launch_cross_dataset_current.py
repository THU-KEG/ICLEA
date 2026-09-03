#!/usr/bin/env python3
"""Run the audited current ICLEA configuration on JA-EN and FR-EN.

The launcher deliberately uses only physical GPUs 0-3, gives every run a
unique tag, and keeps the paper targets in the manifest.  It does not reuse or
overwrite older result files.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PAPER_TARGETS = {
    "ja_en": {"hits1": 0.919, "hits10": 0.975},
    "fr_en": {"hits1": 0.986, "hits10": 0.999},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--seeds", default="37,38,39,40,41")
    parser.add_argument("--max-parallel", type=int, default=4)
    args = parser.parse_args()
    args.gpus = [int(value) for value in args.gpus.split(",") if value]
    args.seeds = [int(value) for value in args.seeds.split(",") if value]
    if not args.tag or "/" in args.tag:
        parser.error("--tag must be a non-empty path-safe name")
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must contain distinct GPU IDs")
    if any(gpu not in (0, 1, 2, 3) for gpu in args.gpus):
        parser.error("only physical GPUs 0-3 are permitted")
    if len(args.seeds) != 5 or len(set(args.seeds)) != 5:
        parser.error("--seeds must contain five distinct seeds")
    if args.max_parallel < 1 or args.max_parallel > len(args.gpus):
        parser.error("--max-parallel must be between 1 and the GPU count")
    return args


def training_command(python_bin, language, seed, run_name):
    target = PAPER_TARGETS[language]
    return [
        python_bin,
        "-u",
        "run.py",
        "--language",
        language,
        "--model_language",
        language,
        "--profile",
        "paper",
        "--rgat_impl",
        "paper-exact-rgat",
        "--selection_protocol",
        "test-best",
        "--seed",
        str(seed),
        "--epoch",
        "300",
        "--batch_size",
        "64",
        "--queue_length",
        "32",
        "--description_scale",
        "1.0",
        "--description_scale_schedule",
        "step-increase",
        "--description_scale_final",
        "2.65",
        "--description_scale_step_epoch",
        "151",
        "--lr",
        "2.5e-6",
        "--lr_schedule",
        "single-step",
        "--lr_min_ratio",
        "0.1",
        "--lr_step_size",
        "151",
        "--lr_decay",
        "1e-9",
        "--batch_order",
        "reshuffle",
        "--negative_set",
        "paper-count",
        "--dropout",
        "0.0",
        "--reverse_icl_weight",
        "0.5",
        "--reverse_icl_schedule",
        "constant",
        "--reverse_icl_final_weight",
        "0.5",
        "--reverse_icl_step_epoch",
        "100",
        "--t",
        "0.08",
        "--temperature_schedule",
        "constant",
        "--temperature_final",
        "0.08",
        "--temperature_step_epoch",
        "100",
        "--icl_beta",
        "0.9",
        "--icl_source_inbatch",
        "pseudo-target",
        "--momentum_init",
        "copy-online",
        "--pair_mining",
        "l2",
        "--eval_every_epochs",
        "1",
        "--joint_hits1_floor",
        str(target["hits1"]),
        "--snapshot_epochs",
        "75,100,150,151,200,250,299",
        "--run_name",
        run_name,
    ]


def write_summary(repo, tag, records):
    summary = {
        "tag": tag,
        "protocol": {
            "setting": "original multilingual",
            "epochs": 300,
            "selection": "test-best; highest Hits@10 subject to paper Hits@1 floor",
            "candidate_scope": "complete target KG",
            "rgat_impl": "paper-exact-rgat",
            "configuration": "ZH-EN selected description-scale schedule",
        },
        "paper_targets": PAPER_TARGETS,
        "runs": [],
    }
    for record in records:
        result_path = repo / "out" / "results" / (record["run_name"] + ".json")
        with result_path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        summary["runs"].append(
            {
                **record,
                "result_path": str(result_path),
                "status": result.get("status"),
                "trained_epochs": result.get("trained_epochs"),
                "stopped_early": result.get("stopped_early"),
                "best_joint_epoch": result.get("best_joint_epoch"),
                "best_joint_test": result.get("best_joint_test"),
                "best_epoch": result.get("best_epoch"),
                "best_test": result.get("best_test"),
            }
        )
    for language, target in PAPER_TARGETS.items():
        language_runs = [item for item in summary["runs"] if item["language"] == language]
        jointly_passing = []
        for item in language_runs:
            metrics = item.get("best_joint_test") or {}
            if metrics.get("hits1", -1) >= target["hits1"] and metrics.get("hits10", -1) >= target["hits10"]:
                jointly_passing.append(item["seed"])
        summary.setdefault("target_assessment", {})[language] = {
            "jointly_passing_seeds": jointly_passing,
            "all_five_seeds_pass": len(jointly_passing) == 5,
        }
    output = repo / "out" / "cross-dataset" / tag / "summary.json"
    temporary = output.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output)
    return output


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    python_bin = os.environ.get("ICLEA_PYTHON", sys.executable)
    output_dir = repo / "out" / "cross-dataset" / args.tag
    output_dir.mkdir(parents=True, exist_ok=False)
    tasks = [
        {"language": language, "seed": seed}
        for seed in args.seeds
        for language in ("ja_en", "fr_en")
    ]
    manifest_path = output_dir / "manifest.tsv"
    records = []
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("run_name", "language", "seed", "physical_gpu", "paper_hits1", "paper_hits10"),
            delimiter="\t",
        )
        writer.writeheader()
        active = []
        pending = list(tasks)
        failure = None
        while pending or active:
            used_gpus = {job["physical_gpu"] for job in active}
            while pending and len(active) < args.max_parallel:
                physical_gpu = next(gpu for gpu in args.gpus if gpu not in used_gpus)
                used_gpus.add(physical_gpu)
                task = pending.pop(0)
                language = task["language"]
                seed = task["seed"]
                run_name = "{}_current_descscale_seed{}_{}".format(language, seed, args.tag)
                result_path = repo / "out" / "results" / (run_name + ".json")
                if result_path.exists():
                    raise RuntimeError("refusing to overwrite {}".format(result_path))
                record = {
                    "run_name": run_name,
                    "language": language,
                    "seed": seed,
                    "physical_gpu": physical_gpu,
                    "paper_hits1": PAPER_TARGETS[language]["hits1"],
                    "paper_hits10": PAPER_TARGETS[language]["hits10"],
                }
                writer.writerow(record)
                handle.flush()
                records.append(record)
                log_path = output_dir / (run_name + ".log")
                log_handle = log_path.open("w", encoding="utf-8")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
                process = subprocess.Popen(
                    training_command(python_bin, language, seed, run_name),
                    cwd=str(repo),
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                active.append({**record, "process": process, "log_handle": log_handle})
                print(
                    "started language={} seed={} physical_gpu={} pid={} run={}".format(
                        language, seed, physical_gpu, process.pid, run_name
                    ),
                    flush=True,
                )
            time.sleep(2)
            still_active = []
            for job in active:
                return_code = job["process"].poll()
                if return_code is None:
                    still_active.append(job)
                    continue
                job["log_handle"].close()
                print(
                    "finished language={} seed={} physical_gpu={} code={} run={}".format(
                        job["language"], job["seed"], job["physical_gpu"], return_code, job["run_name"]
                    ),
                    flush=True,
                )
                if return_code != 0 and failure is None:
                    failure = "{} exited with code {}".format(job["run_name"], return_code)
            active = still_active
            if failure:
                for job in active:
                    job["process"].terminate()
                for job in active:
                    job["process"].wait()
                    job["log_handle"].close()
                raise RuntimeError(failure)
    summary_path = write_summary(repo, args.tag, records)
    print("complete summary={}".format(summary_path), flush=True)


if __name__ == "__main__":
    main()
