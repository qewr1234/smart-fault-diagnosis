"""Cycle projection for a bank-parallel sparse issue core (proposal B, not yet RTL).

Idea: store tap t in tuple sub-list t % T and split each weight bank the same
way. Every cycle each of the T sub-lists issues one tuple, so T tuples times P
lanes multiply per cycle with no bank conflict. A group then costs
max over banks of (nonzero taps in that bank) cycles instead of nnz cycles.
Dense mode costs exactly ceil(K/T) per group.

This script measures that on the real windows: the load imbalance across banks
(the price of a static interleave), and the resulting per-window cycles when
combined with the window-level overlap of overlapped_window_mac (v3), where the
K-beat load is hidden as long as GROUPS*max_bank >= K.

Projection only. It uses the v3 cost structure with the group cost replaced;
no RTL, synthesis or timing exists for this core.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from legacy_run_checks import read_probe  # noqa: E402


def bank_counts(row, t):
    counts = [0]*t
    for i, v in enumerate(row):
        if v:
            counts[i % t] += 1
    return counts


def project(x, k, cout, p, t):
    """Per-window issue cycles for (P lanes, T banks) on window x, sparse mode."""
    groups = (cout+p-1)//p
    counts = bank_counts(x, t)
    per_group = max(counts) if any(counts) else 1
    return groups*per_group, per_group, sum(counts)/t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', default='evaluation')
    ap.add_argument('--P', type=int, nargs='+', default=[2, 8])
    ap.add_argument('--T', type=int, nargs='+', default=[1, 2, 4, 8])
    ap.add_argument('--json', type=Path, default=None)
    args = ap.parse_args()
    x, w, b, ids, y = read_probe(ROOT/'data'/(args.split+'.wpr'))
    k, cout = len(w[0]), len(w)
    nnz = [sum(v != 0 for v in row) for row in x]
    print(f'{args.split}: {len(x)} windows, K={k}, COUT={cout}, mean nnz={st.mean(nnz):.1f} '
          f'({100*(1-st.mean(nnz)/k):.1f}% zero activations)')
    print('\nStatic interleave imbalance (sparse mode, per group):')
    print(f'{"T":>3} {"ideal nnz/T":>12} {"max bank":>9} {"loss":>7}')
    report = dict(split=args.split, K=k, COUT=cout, windows=len(x), mean_nnz=st.mean(nnz), rows=[])
    for t in args.T:
        ideal = st.mean(sum(bank_counts(r, t))/t for r in x)
        worst = st.mean(max(bank_counts(r, t)) for r in x)
        print(f'{t:>3} {ideal:>12.1f} {worst:>9.1f} {100*(worst/ideal-1):>6.1f}%')
    print('\nProjected mean cycles per window, sparse mode (v3 overlap: load hidden when GROUPS*cost >= K):')
    print(f'{"P":>3} {"T":>3} {"issue":>8} {"per window":>11} {"vs v3 T=1":>10} {"DSPs(P*T)":>10}')
    for p in args.P:
        groups = (cout+p-1)//p
        base = None
        for t in args.T:
            issue = st.mean(project(r, k, cout, p, t)[0] for r in x)
            # v3 per-window throughput: issue cycles, plus the K-beat load when it
            # is not hidden, plus ~2 cycles of start handshake per window.
            per_window = st.mean(max(project(r, k, cout, p, t)[0], k)+2 for r in x)
            if base is None:
                base = per_window
            print(f'{p:>3} {t:>3} {issue:>8.1f} {per_window:>11.1f} {100*(1-per_window/base):>9.1f}% {p*t:>10}')
            report['rows'].append(dict(P=p, T=t, mean_issue_cycles=issue, mean_cycles_per_window=per_window,
                                       reduction_vs_T1_pct=100*(1-per_window/base), multipliers=p*t))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        print(f'\nwrote {args.json}')


if __name__ == '__main__':
    main()
