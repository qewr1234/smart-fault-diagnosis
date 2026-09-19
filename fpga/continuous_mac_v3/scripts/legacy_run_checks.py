"""Fresh behavioral RTL runs, integer oracle, and independent cycle comparison.

Python 3.9+ standard library only. Requires Icarus Verilog (iverilog and vvp).
No synthesis, post-route timing, power, full-classifier accuracy or board data.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_vectors(dest, x, w, bias, gold=None, image_ids=None, source=None):
    dest.mkdir(parents=True, exist_ok=True)
    k, c, n = len(w[0]), len(w), len(x)
    assert n > 0 and c > 0 and k > 0
    assert all(len(row) == k and all(0 <= v <= 127 for v in row) for row in x)
    assert all(len(row) == k and all(-128 <= v <= 127 for v in row) for row in w)
    assert all(abs(b)+127*sum(abs(v) for v in row) <= 2**31-1 for b,row in zip(bias,w))
    oracle = [[max(0, b+sum(a*wt for a,wt in zip(row,weights)))
               for b,weights in zip(bias,w)] for row in x]
    if gold is not None:
        assert oracle == gold, "WPROBE gold disagrees with Python integer oracle"
    for name, values, digits in (
        ('weights', (v for row in w for v in row), 2),
        ('bias', iter(bias), 8),
        ('input', (v for row in x for v in row), 2),
        ('gold', (v for row in oracle for v in row), 8)):
        mask = (1 << (digits*4))-1
        (dest/(name+'.hex')).write_text(''.join(f'{v & mask:0{digits}x}\n' for v in values), encoding='ascii')
    meta = dict(K=k,COUT=c,N=n,nonzero=[sum(v != 0 for v in row) for row in x],
                image_ids=image_ids, source=source, integer_oracle_values=n*c,
                sha256={p.name:sha(p) for p in sorted(dest.glob('*.hex'))})
    (dest/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    return meta


def read_probe(path):
    raw = path.read_bytes()
    if raw[:8] != b'WPROBE01':
        raise ValueError('Invalid WPROBE01 magic')
    n,c,k,a = struct.unpack_from('<4I',raw,8)
    if not (0 < n <= 100000 and 0 < c <= 8192 and 0 < k <= 65536 and 0 <= a <= 127):
        raise ValueError('Unsupported WPROBE dimensions/domain')
    if len(raw) != 24+c*k+4*c+4*n+n*k+4*n*c:
        raise ValueError('WPROBE byte count mismatch')
    off = 24
    flat = struct.unpack_from(f'<{c*k}b',raw,off);off += c*k
    w = [flat[i*k:(i+1)*k] for i in range(c)]
    b = struct.unpack_from(f'<{c}i',raw,off);off += c*4
    ids = list(struct.unpack_from(f'<{n}I',raw,off));off += n*4
    x = [list(raw[off+i*k:off+(i+1)*k]) for i in range(n)];off += n*k
    flat = struct.unpack_from(f'<{n*c}i',raw,off)
    y = [list(flat[i*c:(i+1)*c]) for i in range(n)]
    return x,w,b,ids,y


def fixture(k,c):
    rng = random.Random(20260916+k*100+c)
    weights = [[rng.randrange(-128,128) for _ in range(k)] for _ in range(c)]
    weights[0] = [127]*k
    if c > 1:
        weights[1] = [-128]*k
    bias = [17*i-33 for i in range(c)]
    if c > 2:
        weights[2] = [0]*k;bias[2] = 2**31-1
    if c > 3:
        weights[3] = [0]*k;bias[3] = -(2**31-1)
    if c > 4:
        weights[4] = [127]*k;bias[4] = 2**31-1-127*127*k
    if c > 5:
        weights[5] = [-128]*k;bias[5] = -(2**31-1)+127*128*k
    # All nonzero counts for small K, plus alternating and random patterns.
    x = [[0]*k,[127]*k]
    for nz in range(1,k):
        row = [0]*k
        for i in rng.sample(range(k),nz):
            row[i] = rng.randrange(1,128)
        x.append(row)
    x += [[1]*k,[rng.randrange(128) for _ in range(k)],
          [127 if i%2 else 0 for i in range(k)],[0]*k]
    return x,weights,bias


def ready_at(t,stall,pattern,k):
    if pattern == 1:
        return not (t%113 < 17 or t%7 == 0)
    if pattern == 2:
        return t >= k+200 and t%19 >= 7
    return t%16 >= stall


def load_end(k,gaps):
    t = 0
    for _ in range(k):
        while gaps and t%9 in (3,4):
            t += 1
        t += 1
    return t


def cycle_model(k,c,p,n,depth,impl,stall,pattern=0,gaps=0):
    groups = (c+p-1)//p
    def drain(t,lanes):
        for lane in range(lanes):
            while not ready_at(t,stall,pattern,k):
                t += 1
            if lane != lanes-1:
                t += 1
        return t
    start = load_end(k,gaps)
    if impl == 0 or n == 0:
        finish = start
        for g in range(groups):
            first = finish+2 if n == 0 else finish+n+5
            finish = drain(first,min(p,c-g*p))
        return finish
    finishes = []
    issue_start = start+2
    for g in range(groups):
        if g:
            issue_start += n
        if g >= depth:
            issue_start = max(issue_start,finishes[g-depth])
        first = max(issue_start+n+3,finishes[-1]+1 if finishes else 0)
        finishes.append(drain(first,min(p,c-g*p)))
    return finishes[-1]


def command(args,cwd,log,timeout=1800):
    proc = subprocess.run([str(x) for x in args],cwd=cwd,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,text=True,timeout=timeout)
    log.write_text(proc.stdout,encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(f'Command failed ({proc.returncode}): {log}\n{proc.stdout[-6000:]}')
    return proc.stdout


def run_case(case,out,vectors,iverilog,vvp):
    name,vec,impl,p,depth,stall,pattern,gaps,reset = case
    dest = out/name;dest.mkdir()
    meta = vectors[vec]
    params = dict(K=meta['K'],COUT=meta['COUT'],P=p,DEPTH=depth,N=meta['N'],
                  IMPL=impl,STALL_LEN=stall,PATTERN=pattern,INPUT_GAPS=gaps,RESET_TEST=reset)
    sources = [out/'sources/rtl/sparse_window_mac.sv',out/'sources/rtl/continuous_window_mac.sv',out/'sources/sim/tb_window_compare.sv']
    argv = [iverilog,'-g2012','-Wall','-s','tb_window_compare','-o',dest/'sim.vvp']
    argv += [f'-Ptb_window_compare.{key}={value}' for key,value in params.items()]
    command(argv+sources,dest,dest/'compile.log')
    log = command([vvp,dest/'sim.vvp','+VEC='+str((out/'vectors'/vec).resolve())],dest,dest/'simulation.log')
    match = re.search(r'PASS ALL checked_values=(\d+)',log)
    assert match and int(match.group(1)) == meta['N']*meta['COUT']*2, log
    assert 'FATAL' not in log and 'ERROR' not in log, log
    with (dest/'rtl_cycles.csv').open(newline='') as f:
        rows = [{key:int(v) for key,v in row.items()} for row in csv.DictReader(f)]
    assert len(rows) == meta['N']*2
    for i,row in enumerate(rows):
        frame,mode = divmod(i,2)
        nz = meta['nonzero'][frame]
        assert row['frame'] == frame and row['sparse_mode'] == mode
        assert row['nonzero_inputs'] == nz
        n = nz if mode else meta['K']
        expect = cycle_model(meta['K'],meta['COUT'],p,n,depth,impl,stall,pattern,gaps)
        assert row['cycles'] == expect, (name,frame,mode,row['cycles'],expect)
        assert row['load_end_cycle'] == load_end(meta['K'],gaps)
        assert row['issue_steps'] == ((meta['COUT']+p-1)//p)*n
    report = dict(name=name,vector=vec,**params,checked_values=int(match.group(1)),
                  cycle_rows=len(rows),cycle_model_mismatches=0,
                  mean_dense_cycles=sum(r['cycles'] for r in rows[0::2])/meta['N'],
                  mean_sparse_cycles=sum(r['cycles'] for r in rows[1::2])/meta['N'])
    print(f'PASS {name}: values={report["checked_values"]}, cycles agree ({len(rows)} rows)',flush=True)
    (dest/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite',choices=['smoke','full'],default='smoke')
    ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--iverilog',default=os.environ.get('IVERILOG','iverilog'))
    ap.add_argument('--vvp',default=os.environ.get('VVP','vvp'))
    args = ap.parse_args()
    iverilog,vvp = shutil.which(args.iverilog),shutil.which(args.vvp)
    if not iverilog or not vvp:
        raise SystemExit('Icarus Verilog not found. Put iverilog and vvp on PATH, or pass --iverilog / --vvp. See README_KO.md.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    out = ROOT/'build'/('rtl_'+stamp);out.mkdir(parents=True)
    print(f'Fresh RTL run: {out}',flush=True)
    # Freeze the exact executable inputs before launching any concurrent cases.
    sources = ['rtl/sparse_window_mac.sv','rtl/continuous_window_mac.sv',
               'sim/tb_window_compare.sv','scripts/run_checks.py']
    source_hashes = {}
    for relative in sources:
        saved = out/'sources'/relative
        saved.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,saved)
        source_hashes[relative] = sha(saved)
    versions = {}
    for name,exe in [('iverilog',iverilog),('vvp',vvp)]:
        versions[name] = command([exe,'-V'],out,out/(name+'_version.txt'))
    vectors = {}
    for k,c in [(1,1),(1,9),(9,9),(9,17),(27,9)]:
        name = f'synthetic_k{k}_c{c}'
        x,w,b = fixture(k,c)
        vectors[name] = export_vectors(out/'vectors'/name,x,w,b,source={'synthetic_seed':20260916+k*100+c})
    cases = []
    for p in (1,2,4,8):
        for depth in (1,2,3):
            for stall in (0,12):
                name = f'v2_small_p{p}_d{depth}_s{stall}'
                cases.append((name,'synthetic_k9_c9',1,p,depth,stall,0,0,0))
    for k,c,p,depth,stall in [(1,1,1,1,0),(1,1,8,2,12),(1,9,1,4,0),
                             (1,9,3,2,12),(9,17,8,2,4),(9,9,3,3,4),
                             (9,9,8,8,12)]:
        cases.append((f'v2_edge_k{k}_c{c}_p{p}_d{depth}_s{stall}',f'synthetic_k{k}_c{c}',1,p,depth,stall,0,0,0))
    for impl in (0,1):
        for pattern in (0,1,2):
            cases.append((f'v{impl+1}_gaps_pattern{pattern}','synthetic_k27_c9',impl,4,2,12,pattern,1,1))
        for p in (1,8):
            cases.append((f'v{impl+1}_minimum_p{p}','synthetic_k1_c9',impl,p,2,12,0,0,1))
    if args.suite == 'full':
        for split in ('calibration','evaluation'):
            path = ROOT/'data'/(split+'.wpr')
            x,w,b,ids,y = read_probe(path)
            vectors[split] = export_vectors(out/'vectors'/split,x,w,b,y,ids,
                                            {'file':split+'.wpr','sha256':sha(path),'selection':'all 512 recorded windows'})
            for impl in (0,1):
                cases.append((f'v{impl+1}_{split}_p2_d2_s0',split,impl,2,2,0,0,0,0))
        for impl in (0,1):
            for stall in (0,12):
                cases.append((f'v{impl+1}_evaluation_p8_d2_s{stall}','evaluation',impl,8,2,stall,0,0,0))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,min(4,args.jobs))) as pool:
        futures = [pool.submit(run_case,case,out,vectors,iverilog,vvp) for case in cases]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    results.sort(key=lambda r:r['name'])
    report = dict(pass_all=True,runtime='Icarus Verilog behavioral RTL simulation',
                  utc=datetime.now(timezone.utc).isoformat(),suite=args.suite,
                  test_cases=len(results),checked_rtl_outputs=sum(r['checked_values'] for r in results),
                  checked_cycle_rows=sum(r['cycle_rows'] for r in results),
                  mismatches=0,cycle_model_mismatches=0,fpga_synthesis_performed=False,
                  board_measurement_performed=False,new_rtl_simulation_performed=True,
                  source_sha256=source_hashes,
                  cases=results)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    print(f'PASS ALL: {len(results)} RTL cases, {report["checked_rtl_outputs"]} outputs, {report["checked_cycle_rows"]} cycle rows.',flush=True)
    print('FPGA synthesis / route / board measurements: NOT performed.',flush=True)
    print(f'Results: {out}',flush=True)


if __name__ == '__main__':
    main()
