"""Back-to-back window stream: RTL runs, integer oracle, and a second cycle model.

Runs tb_stream_compare.sv on v1 (sparse_window_mac), v2 (continuous_window_mac)
and v3 (overlapped_window_mac) with the same window stream, checks every output
value against the Python integer oracle, and checks every window's start/end
edge against an independent per-edge protocol model of each core (stream_model).

Python 3.9+ standard library only. Requires Icarus Verilog (iverilog and vvp).
No synthesis, post-route timing, power, full-classifier accuracy or board data.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from legacy_run_checks import export_vectors, fixture, read_probe, ready_at, sha  # noqa: E402
from stream_model import stream_model  # noqa: E402


def command(args, cwd, log, timeout=7200):
    proc = subprocess.run([str(x) for x in args], cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=timeout)
    log.write_text(proc.stdout, encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(f'Command failed ({proc.returncode}): {log}\n{proc.stdout[-6000:]}')
    return proc.stdout


def windows_of(meta, mode_seq):
    n = meta['N']
    if mode_seq == 2:
        return [(w//2, w % 2) for w in range(2*n)]
    return [(w, mode_seq) for w in range(n)]


def run_case(case, out, vectors, iverilog, vvp):
    name, vec, impl, p, depth, stall, pattern, gaps, mode_seq, reset = case
    dest = out/name
    dest.mkdir()
    meta = vectors[vec]
    params = dict(K=meta['K'], COUT=meta['COUT'], P=p, DEPTH=depth, N=meta['N'], IMPL=impl,
                  STALL_LEN=stall, PATTERN=pattern, INPUT_GAPS=gaps, MODE_SEQ=mode_seq, RESET_TEST=reset)
    sources = [out/'sources/rtl/sparse_window_mac.sv', out/'sources/rtl/continuous_window_mac.sv',
               out/'sources/rtl/overlapped_window_mac.sv', out/'sources/sim/tb_stream_compare.sv']
    argv = [iverilog, '-g2012', '-Wall', '-s', 'tb_stream_compare', '-o', dest/'sim.vvp']
    argv += [f'-Ptb_stream_compare.{key}={value}' for key, value in params.items()]
    command(argv+sources, dest, dest/'compile.log')
    log = command([vvp, dest/'sim.vvp', '+VEC='+str((out/'vectors'/vec).resolve())], dest, dest/'simulation.log')
    match = re.search(r'PASS ALL checked_values=(\d+) windows=(\d+) total_cycles=(\d+)', log)
    wins = windows_of(meta, mode_seq)
    assert match and int(match.group(1)) == len(wins)*meta['COUT'], log
    assert int(match.group(2)) == len(wins), log
    assert 'FATAL' not in log and 'ERROR' not in log, log
    with (dest/'rtl_cycles.csv').open(newline='') as f:
        rows = [{key: int(v) for key, v in row.items()} for row in csv.DictReader(f)]
    assert len(rows) == len(wins)
    nz = meta['nonzero']
    taps = [nz[frame] if mode else meta['K'] for frame, mode in wins]
    model = stream_model(meta['K'], meta['COUT'], p, depth, impl, taps, stall, pattern, gaps)
    t0 = rows[0]['start_cycle']
    for w, (row, (frame, mode)) in enumerate(zip(rows, wins)):
        assert row['window'] == w and row['frame'] == frame and row['sparse_mode'] == mode
        assert row['nonzero_inputs'] == nz[frame]
        got = (row['start_cycle']-t0, row['end_cycle']-t0)
        assert got == model[w], (name, w, frame, mode, got, model[w])
    total = rows[-1]['end_cycle']-t0
    dense = [r['end_cycle']-r['start_cycle'] for r, (_, m) in zip(rows, wins) if m == 0]
    sparse = [r['end_cycle']-r['start_cycle'] for r, (_, m) in zip(rows, wins) if m == 1]
    report = dict(name=name, vector=vec, **params, checked_values=int(match.group(1)),
                  windows=len(wins), cycle_rows=len(rows), cycle_model_mismatches=0,
                  total_cycles=total, cycles_per_window=total/len(wins),
                  mean_dense_latency=(sum(dense)/len(dense) if dense else None),
                  mean_sparse_latency=(sum(sparse)/len(sparse) if sparse else None))
    print(f'PASS {name}: values={report["checked_values"]}, {len(wins)} windows, '
          f'{total} cycles ({total/len(wins):.1f}/window), model agrees', flush=True)
    (dest/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite', choices=['smoke', 'full'], default='smoke')
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--iverilog', default=os.environ.get('IVERILOG', 'iverilog'))
    ap.add_argument('--vvp', default=os.environ.get('VVP', 'vvp'))
    args = ap.parse_args()
    iverilog, vvp = shutil.which(args.iverilog), shutil.which(args.vvp)
    if not iverilog or not vvp:
        raise SystemExit('Icarus Verilog not found. Put iverilog and vvp on PATH, or pass --iverilog / --vvp.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    out = ROOT/'build'/('stream_'+stamp)
    out.mkdir(parents=True)
    print(f'Fresh RTL run: {out}', flush=True)
    sources = ['rtl/sparse_window_mac.sv', 'rtl/continuous_window_mac.sv', 'rtl/overlapped_window_mac.sv',
               'sim/tb_stream_compare.sv', 'scripts/run_stream_checks.py', 'scripts/stream_model.py',
               'scripts/legacy_run_checks.py']
    source_hashes = {}
    for relative in sources:
        saved = out/'sources'/relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, saved)
        source_hashes[relative] = sha(saved)
    versions = {}
    for name, exe in [('iverilog', iverilog), ('vvp', vvp)]:
        versions[name] = command([exe, '-V'], out, out/(name+'_version.txt'))
    vectors = {}
    for k, c in [(1, 1), (1, 9), (9, 9), (9, 17), (27, 9)]:
        name = f'synthetic_k{k}_c{c}'
        x, w, b = fixture(k, c)
        vectors[name] = export_vectors(out/'vectors'/name, x, w, b, source={'synthetic_seed': 20260916+k*100+c})
    cases = []
    # (name, vector, impl, P, DEPTH, stall, pattern, gaps, mode_seq, reset)
    for impl in (0, 1, 2):
        for p in (1, 2, 4, 8):
            for depth in (1, 2, 3):
                for stall in (0, 12):
                    cases.append((f'v{impl+1}_small_p{p}_d{depth}_s{stall}', 'synthetic_k9_c9', impl, p, depth, stall, 0, 0, 2, 0))
    for k, c, p, depth, stall in [(1, 1, 1, 1, 0), (1, 1, 8, 2, 12), (1, 9, 1, 4, 0), (1, 9, 3, 2, 12),
                                  (9, 17, 8, 2, 4), (9, 9, 3, 3, 4), (9, 9, 8, 8, 12)]:
        cases.append((f'v3_edge_k{k}_c{c}_p{p}_d{depth}_s{stall}', f'synthetic_k{k}_c{c}', 2, p, depth, stall, 0, 0, 2, 0))
    for impl in (0, 1, 2):
        for pattern in (0, 1, 2):
            cases.append((f'v{impl+1}_gaps_pattern{pattern}', 'synthetic_k27_c9', impl, 4, 2, 12, pattern, 1, 2, 1))
        for p in (1, 8):
            cases.append((f'v{impl+1}_minimum_p{p}', 'synthetic_k1_c9', impl, p, 2, 12, 0, 0, 2, 1))
        for mode_seq in (0, 1):
            cases.append((f'v{impl+1}_modeseq{mode_seq}_p4_d2', 'synthetic_k27_c9', impl, 4, 2, 0, 0, 0, mode_seq, 0))
    if args.suite == 'full':
        for split in ('calibration', 'evaluation'):
            path = ROOT/'data'/(split+'.wpr')
            x, w, b, ids, y = read_probe(path)
            vectors[split] = export_vectors(out/'vectors'/split, x, w, b, y, ids,
                                            {'file': split+'.wpr', 'sha256': sha(path), 'selection': 'all 512 recorded windows'})
            for impl in (0, 1, 2):
                cases.append((f'v{impl+1}_{split}_p2_d2_s0', split, impl, 2, 2, 0, 0, 0, 2, 0))
        for impl in (0, 1, 2):
            for stall in (0, 12):
                cases.append((f'v{impl+1}_evaluation_p8_d2_s{stall}', 'evaluation', impl, 8, 2, stall, 0, 0, 2, 0))
        for p in (16, 32):
            for impl in (1, 2):
                cases.append((f'v{impl+1}_evaluation_p{p}_d2_s0', 'evaluation', impl, p, 2, 0, 0, 0, 2, 0))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(run_case, case, out, vectors, iverilog, vvp) for case in cases]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    results.sort(key=lambda r: r['name'])
    report = dict(pass_all=True, runtime='Icarus Verilog behavioral RTL simulation',
                  utc=datetime.now(timezone.utc).isoformat(), suite=args.suite,
                  test_cases=len(results), checked_rtl_outputs=sum(r['checked_values'] for r in results),
                  checked_cycle_rows=sum(r['cycle_rows'] for r in results),
                  mismatches=0, cycle_model_mismatches=0, fpga_synthesis_performed=False,
                  board_measurement_performed=False, new_rtl_simulation_performed=True,
                  source_sha256=source_hashes, cases=results)
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    with (out/'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f'PASS ALL: {len(results)} RTL cases, {report["checked_rtl_outputs"]} outputs, {report["checked_cycle_rows"]} cycle rows.', flush=True)
    print('FPGA synthesis / route / board measurements: NOT performed.', flush=True)
    print(f'Results: {out}', flush=True)


if __name__ == '__main__':
    main()
