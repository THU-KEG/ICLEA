#!/usr/bin/env python3
"""Run five independent ICLEA seeds and aggregate best monitored-test metrics."""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', choices=('zh_en', 'ja_en', 'fr_en'), required=True)
    parser.add_argument('--setting', choices=('original', 'translated'), default='original')
    parser.add_argument('--profile', choices=('paper', 'top1-no-threshold'), default='paper')
    parser.add_argument('--seeds', default='37,38,39,40,41')
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--max-parallel', type=int, default=4)
    parser.add_argument('--reuse-completed', action='store_true')
    parser.add_argument('training_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.seeds = [int(value) for value in args.seeds.split(',') if value]
    args.gpus = [int(value) for value in args.gpus.split(',') if value]
    if len(args.seeds) != 5:
        parser.error('--seeds must contain exactly five seeds')
    if len(set(args.seeds)) != 5:
        parser.error('--seeds must be five distinct seeds for a meaningful standard deviation')
    if not args.gpus or any(gpu not in (0, 1, 2, 3) for gpu in args.gpus):
        parser.error('--gpus may contain only physical GPU IDs 0,1,2,3')
    if len(set(args.gpus)) != len(args.gpus):
        parser.error('--gpus contains a duplicate ID')
    args.max_parallel = min(args.max_parallel, len(args.gpus), 4)
    if args.max_parallel < 1:
        parser.error('--max-parallel must be positive')
    if args.training_args and args.training_args[0] == '--':
        args.training_args = args.training_args[1:]
    return args


def aggregate(result_paths, output_path, metadata):
    runs = []
    for path in result_paths:
        with path.open(encoding='utf-8') as f:
            result = json.load(f)
        if result.get('status') != 'complete':
            raise RuntimeError('{} is not a completed run'.format(path))
        protocol = result.get('protocol', {})
        if protocol.get('selection_protocol') != 'test_best':
            raise RuntimeError('{} is not a test-best run'.format(path))
        runs.append(result)

    def stats(key):
        values = [float(run['best_test'][key]) for run in runs]
        return {
            'values': values,
            'mean': statistics.mean(values),
            'sample_std': statistics.stdev(values),
        }

    report = dict(metadata)
    report.update({
        'num_runs': len(runs),
        'selection_protocol': 'test_best',
        'selection_metric': 'test_hits1_then_hits10',
        'standard_deviation': 'sample (N-1 denominator)',
        'best_test_hits1': stats('hits1'),
        'best_test_hits10': stats('hits10'),
        'best_epochs': [run['best_epoch'] for run in runs],
        'run_files': [str(path) for path in result_paths],
    })
    temporary = output_path.with_suffix(output_path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write('\n')
    temporary.replace(output_path)
    return report


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    result_dir = repo / 'out' / 'results'
    log_dir = repo / 'out' / 'five-run-logs'
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    setting_arg = ['--trans'] if args.setting == 'translated' else []
    pending = list(args.seeds)
    active = []
    result_paths = []

    while pending or active:
        while pending and len(active) < args.max_parallel:
            seed = pending.pop(0)
            used_gpus = {job[2] for job in active}
            gpu = next(gpu_id for gpu_id in args.gpus if gpu_id not in used_gpus)
            run_name = '{}_{}_{}_testbest_seed{}'.format(
                args.language, args.setting, args.profile, seed
            )
            result_path = result_dir / (run_name + '.json')
            if args.reuse_completed and result_path.exists():
                with result_path.open(encoding='utf-8') as f:
                    if json.load(f).get('status') == 'complete':
                        result_paths.append(result_path)
                        continue
            log_path = log_dir / (run_name + '.log')
            command = [
                sys.executable,
                '-u',
                'run.py',
                '--language', args.language,
                '--model_language', args.language,
                '--profile', args.profile,
                '--selection_protocol', 'test-best',
                '--seed', str(seed),
                '--run_name', run_name,
            ] + setting_arg + args.training_args
            environment = os.environ.copy()
            environment['CUDA_VISIBLE_DEVICES'] = str(gpu)
            log_file = log_path.open('w', encoding='utf-8')
            process = subprocess.Popen(
                command,
                cwd=str(repo),
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            print('started seed={} physical_gpu={} pid={} log={}'.format(
                seed, gpu, process.pid, log_path), flush=True)
            active.append((process, seed, gpu, result_path, log_file))

        time.sleep(1)
        still_active = []
        for process, seed, gpu, result_path, log_file in active:
            return_code = process.poll()
            if return_code is None:
                still_active.append((process, seed, gpu, result_path, log_file))
                continue
            log_file.close()
            if return_code != 0:
                raise RuntimeError('seed {} exited with code {}'.format(seed, return_code))
            if not result_path.exists():
                raise RuntimeError('seed {} produced no result JSON'.format(seed))
            result_paths.append(result_path)
            print('completed seed={} result={}'.format(seed, result_path), flush=True)
        active = still_active

    result_paths.sort(key=lambda path: args.seeds.index(int(path.stem.rsplit('seed', 1)[1])))
    report_path = result_dir / '{}_{}_{}_testbest_five_run_summary.json'.format(
        args.language, args.setting, args.profile
    )
    report = aggregate(
        result_paths,
        report_path,
        {
            'language': args.language,
            'setting': args.setting,
            'profile': args.profile,
            'seeds': args.seeds,
            'physical_gpus': args.gpus,
        },
    )
    print('FIVE_RUN_SUMMARY_JSON=' + json.dumps(report, sort_keys=True))
    print('Summary:', report_path)


if __name__ == '__main__':
    main()
