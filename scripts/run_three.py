#!/usr/bin/env python3
"""Run five test-best seeds for all three DBP15K language pairs on GPUs 0-3."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_five import aggregate


LANGUAGES = ('zh_en', 'ja_en', 'fr_en')


def parse_args():
    parser = argparse.ArgumentParser()
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
    if len(args.seeds) != 5 or len(set(args.seeds)) != 5:
        parser.error('--seeds must contain exactly five distinct seeds')
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


def completed_testbest(path):
    if not path.exists():
        return False
    with path.open(encoding='utf-8') as f:
        result = json.load(f)
    return (
        result.get('status') == 'complete'
        and result.get('protocol', {}).get('selection_protocol') == 'test_best'
    )


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    result_dir = repo / 'out' / 'results'
    log_dir = repo / 'out' / 'three-run-logs'
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    setting_arg = ['--trans'] if args.setting == 'translated' else []
    pending = [(language, seed) for language in LANGUAGES for seed in args.seeds]
    active = []
    result_paths = {language: [] for language in LANGUAGES}

    while pending or active:
        while pending and len(active) < args.max_parallel:
            language, seed = pending.pop(0)
            used_gpus = {job[3] for job in active}
            gpu = next(gpu_id for gpu_id in args.gpus if gpu_id not in used_gpus)
            run_name = '{}_{}_{}_testbest_seed{}'.format(
                language, args.setting, args.profile, seed
            )
            result_path = result_dir / (run_name + '.json')
            if args.reuse_completed and completed_testbest(result_path):
                result_paths[language].append(result_path)
                print('reused language={} seed={} result={}'.format(
                    language, seed, result_path
                ), flush=True)
                continue
            log_path = log_dir / (run_name + '.log')
            command = [
                sys.executable,
                '-u',
                'run.py',
                '--language', language,
                '--model_language', language,
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
            print('started language={} seed={} physical_gpu={} pid={} log={}'.format(
                language, seed, gpu, process.pid, log_path
            ), flush=True)
            active.append((process, language, seed, gpu, result_path, log_file))

        time.sleep(1)
        still_active = []
        for process, language, seed, gpu, result_path, log_file in active:
            return_code = process.poll()
            if return_code is None:
                still_active.append(
                    (process, language, seed, gpu, result_path, log_file)
                )
                continue
            log_file.close()
            if return_code != 0:
                raise RuntimeError(
                    '{} seed {} exited with code {}'.format(language, seed, return_code)
                )
            if not completed_testbest(result_path):
                raise RuntimeError(
                    '{} seed {} produced no completed test-best JSON'.format(language, seed)
                )
            result_paths[language].append(result_path)
            print('completed language={} seed={} result={}'.format(
                language, seed, result_path
            ), flush=True)
        active = still_active

    for language in LANGUAGES:
        paths = result_paths[language]
        paths.sort(key=lambda path: args.seeds.index(int(path.stem.rsplit('seed', 1)[1])))
        report_path = result_dir / '{}_{}_{}_testbest_five_run_summary.json'.format(
            language, args.setting, args.profile
        )
        report = aggregate(
            paths,
            report_path,
            {
                'language': language,
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
