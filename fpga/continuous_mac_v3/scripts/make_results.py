"""Summarise a run_stream_checks build directory into a v1/v2/v3 comparison table.

Usage: python scripts/make_results.py build/stream_<stamp> [--json out.json]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('build', type=Path)
    ap.add_argument('--json', type=Path, default=None)
    args = ap.parse_args()
    summary = json.loads((args.build/'summary.json').read_text(encoding='utf-8'))
    by_cfg = defaultdict(dict)
    for case in summary['cases']:
        if not case['vector'] in ('calibration', 'evaluation'):
            continue
        key = (case['vector'], case['P'], case['DEPTH'], case['STALL_LEN'])
        by_cfg[key][case['IMPL']] = case
    rows = []
    print(f'{"data":<12}{"P":>3}{"D":>3}{"stall":>6} | {"v1 serial":>10}{"v2 cont.":>10}{"v3 overlap":>11} | '
          f'{"v3 vs v2":>9}{"v3 vs v1":>9} | {"v3 dense lat":>13}{"v3 sparse lat":>14}')
    for key in sorted(by_cfg):
        cases = by_cfg[key]
        per = {impl: c['cycles_per_window'] for impl, c in cases.items()}
        v3 = cases.get(2)
        row = dict(vector=key[0], P=key[1], DEPTH=key[2], stall=key[3],
                   cycles_per_window={f'v{i+1}': per[i] for i in sorted(per)},
                   windows=next(iter(cases.values()))['windows'])
        if v3:
            row['v3_mean_dense_latency'] = v3['mean_dense_latency']
            row['v3_mean_sparse_latency'] = v3['mean_sparse_latency']
        if 1 in per and 2 in per:
            row['v3_vs_v2_pct'] = 100*(1-per[2]/per[1])
        if 0 in per and 2 in per:
            row['v3_vs_v1_pct'] = 100*(1-per[2]/per[0])
        rows.append(row)
        f = lambda v: f'{v:>10.1f}' if v is not None else f'{"-":>10}'
        print(f'{key[0]:<12}{key[1]:>3}{key[2]:>3}{key[3]:>6} | {f(per.get(0))}{f(per.get(1))}{f(per.get(2)):>11} | '
              f'{row.get("v3_vs_v2_pct", float("nan")):>8.2f}%{row.get("v3_vs_v1_pct", float("nan")):>8.2f}% | '
              f'{(v3 or {}).get("mean_dense_latency") or float("nan"):>13.1f}{(v3 or {}).get("mean_sparse_latency") or float("nan"):>14.1f}')
    print('\ncycles per window = stream total / (512 windows x 2 modes); dense and sparse windows alternate in the stream.')
    if args.json:
        args.json.write_text(json.dumps(dict(build=str(args.build), utc=summary['utc'], rows=rows), indent=2)+'\n', encoding='utf-8')
        print(f'wrote {args.json}')


if __name__ == '__main__':
    main()
