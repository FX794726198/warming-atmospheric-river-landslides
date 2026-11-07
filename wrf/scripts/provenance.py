"""Final provenance of the 10 AR PGW runs, built from the output that the post-processing actually uses.
usage: provenance_v2.py [RUN ...] > provenance_final.md        (default: all 10 runs)

For each run, the d02 hourly (wrfaux5) and d01 3-hourly (wrfaux7) frames are selected exactly as post_case.py v2 does (one frame
per output time; a time written by two segments is taken from the file written last, the segment that continued the run).
Each kept frame is traced to the file that holds it. The file's global attributes give
  START_DATE  the model time the writing segment started from (cold start or the restart it read),
  DT          the time step the segment ran with,
  physics     every other global attribute except the segment start time, the event geography (projection centre, true
              latitudes, nest position, grid size, simulation start) and the MPI decomposition (NTASKS_*, *_PATCH_*): physics,
              dynamics, diffusion, Noah-MP options, vertical levels, nesting, WRF version; hashed per domain;
and its modification time is matched to the job segment that wrote it ("wrf start" .. "wrf stop" in the Slurm logs, else the
job's sacct run time). Jobs whose output is not used anywhere (stalled, cancelled, retried, superseded) are listed per run.
Checks per run: gaps in the kept series, duplicated times (and whether the copies are bit-identical, from dupcheck logic), and
non-finite values in RAINNC, RAINC, T2 (d02) and RAINNC, PSFC (d01) of the kept frames."""
import os, re, sys, glob, time, hashlib, subprocess, collections, datetime as dt, numpy as np, netCDF4
B = os.path.expandvars("/scratch/$USER/ar_pgw")
sys.path.insert(0, f"{B}/scripts"); from events import EVENTS
RUNS = sys.argv[1:] or [f"{e}_{a}" for e in ("NOR_2011", "ORE_1996", "RSB_2024", "BC_2021", "LIG_2014") for a in ("ctl", "pgw585")]
SKIP_ATTR = {"START_DATE", "JULYR", "JULDAY", "GMT",                                   # segment start time
             "CEN_LAT", "CEN_LON", "MOAD_CEN_LAT", "STAND_LON", "TRUELAT1", "TRUELAT2", "POLE_LAT", "POLE_LON",   # event geography
             "I_PARENT_START", "J_PARENT_START", "SIMULATION_START_DATE", "WEST-EAST_GRID_DIMENSION", "SOUTH-NORTH_GRID_DIMENSION"}
def skip_attr(k): return k in SKIP_ATTR or "_PATCH_" in k or k.startswith("NTASKS")   # MPI decomposition of the segment
run_start = {r: EVENTS[r.rsplit("_", 1)[0]]["start"] for r in RUNS}
ORE_RERUN = time.mktime(time.strptime("2026-11-05 07:43", "%Y-%m-%d %H:%M"))   # 06:43Z: ORE ctl dt-45 rerun set up; earlier ORE ctl jobs = dt36_mixed
def tstr(ds): return ["".join(t.astype(str)) for t in ds["Times"][:]]
def epoch_local(s):   # "Thu Nov  5 15:33:19 CET 2026" (login/compute nodes run Europe/Amsterdam)
    s = re.sub(r"\s+[A-Z]{3,4}\s+(\d{4})$", r" \1", s.strip()); return time.mktime(time.strptime(s, "%a %b %d %H:%M:%S %Y"))
_sacct = {}
def sacct(j):
    if j not in _sacct:
        o = subprocess.run(["sacct", "-n", "-X", "-P", "-j", j, "-o", "Partition,NNodes,NCPUS,State,Start,End"], capture_output=True, text=True).stdout.strip().splitlines()
        f = o[0].split("|") if o else ["?"] * 6
        conv = lambda s: time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%S")) if re.match(r"\d{4}-", s) else None
        _sacct[j] = dict(part=f[0], nodes=f[1], cpus=f[2], state=f[3].split()[0] if f[3] else "?", start=conv(f[4]), end=conv(f[5]) or time.time())
    return _sacct[j]
# ---- segments from the Slurm logs: (run, job, mode, from, wrf start epoch, wrf stop epoch or None, rc, note)
SEG = collections.defaultdict(list)
for f in glob.glob(f"{B}/logs/wrf_*.log") + glob.glob(f"{B}/logs/lane_ar_innov_*.log"):
    jid = re.search(r"_(\d+)\.log$", f).group(1); txt = open(f, errors="ignore").read()
    for m in re.finditer(r"job (\d+) (\w+_\d{4}) (ctl|pgw585) mode (\w+)(.*?)(?=\njob \d+ \w+_\d{4} (?:ctl|pgw585) mode|\Z)", txt, re.S):
        run = f"{m.group(2)}_{m.group(3)}"; body = m.group(5)
        for w in re.finditer(r"(restart from (\S+)|cold start)\n(?:.*\n)*?wrf start (.*?), stops after.*?\n(.*?)(?=(?:restart from \S+|cold start)\n|\Z)", body, re.S):
            stop = re.search(r"wrf stop (.*?) rc (\d+) wall", w.group(4))
            SEG[run].append(dict(job=jid, mode=m.group(4), frm=(w.group(2) or "cold")[:13], t0=epoch_local(w.group(3)),
                                 t1=epoch_local(stop.group(1)) if stop else None, rc=stop.group(2) if stop else None,
                                 ok="run complete" in w.group(4), retry36="retrying once with time_step 36" in w.group(4)))
def owner(run, mtime, start):
    """job segment of this run that started from the file's START_DATE and was running when the file was last written
    (any segment running at that time if none matches the start)"""
    frm = "cold" if start == run_start[run] else start
    cand = [s for s in SEG.get(run, []) if s["frm"] == frm] or SEG.get(run, [])
    best = None
    for s in cand:
        t1 = s["t1"] or sacct(s["job"])["end"]
        if s["t0"] - 60 <= mtime <= t1 + 300 and (best is None or s["t0"] > best["t0"]): best = s
    return best
def kept(run, pattern):
    pick = {}
    for f in sorted(glob.glob(f"{B}/runs/{run}/{pattern}"), key=lambda f: (os.path.getmtime(f), f)):
        with netCDF4.Dataset(f) as ds:
            for i, t in enumerate(tstr(ds)): pick.setdefault(t, []).append((f, i))
    return {t: L[-1] for t, L in pick.items()}, {t: L for t, L in pick.items() if len(L) > 1}
FP = {}
rows, checks, unused = [], [], {}
for run in RUNS:
    used_jobs = set()
    for dom, pat, step, nanv in (("d02", "wrfaux5_d02_*", 1, ("RAINNC", "RAINC", "T2")), ("d01", "wrfaux7_d01_*", 3, ("RAINNC", "PSFC"))):
        K, dups = kept(run, pat)
        T = sorted(K); TT = [dt.datetime.strptime(t, "%Y-%m-%d_%H:%M:%S") for t in T]
        gaps = [str(TT[i - 1] + dt.timedelta(hours=step))[:13] for i in range(1, len(TT)) if (TT[i] - TT[i - 1]).total_seconds() != step * 3600]
        ident = 0
        for t, L in dups.items():
            (f1, i1), (f2, i2) = L[0], L[-1]; same = True
            with netCDF4.Dataset(f1) as a, netCDF4.Dataset(f2) as b:
                for v in a.variables:
                    if v != "Times" and v in b.variables and not np.array_equal(np.asarray(a[v][i1]), np.asarray(b[v][i2]), equal_nan=True): same = False; break
            ident += same
        nonfinite = 0; groups = collections.OrderedDict()
        for t in T:
            f, i = K[t]; groups.setdefault(f, []).append(t)
        for f, ts in groups.items():
            with netCDF4.Dataset(f) as ds:
                a = ds.__dict__; idx = [tstr(ds).index(t) for t in ts]
                for v in nanv:
                    if v in ds.variables: nonfinite += int((~np.isfinite(np.asarray(ds[v][idx], "f8"))).sum())
                fp = hashlib.md5(repr(sorted((k, str(a[k])) for k in a if not skip_attr(k) and k != "DT")).encode()).hexdigest()[:8]
                FP.setdefault(dom, set()).add(fp)
                s = owner(run, os.path.getmtime(f), a["START_DATE"][:13]); job = s["job"] if s else "?"; used_jobs.add(job)
                rows.append(dict(run=run, dom=dom, start=a["START_DATE"][:13], dt=float(a["DT"]), fp=fp, job=job, seg=s, first=ts[0][:13], last=ts[-1][:13], n=len(ts)))
        checks.append(dict(run=run, dom=dom, n=len(T), first=T[0][:13], last=T[-1][:13], gaps=gaps, dups=len(dups), ident=ident, nonfinite=nonfinite))
    unused[run] = [s for s in SEG.get(run, []) if s["job"] not in used_jobs]
# ---- compact table, one row per run (added 2026-11-05 for the final PLAN.md table)
def physid(nl):   # same as provenance.py: &physics, &noah_mp, &dynamics, &bdy_control + e_vert / p_top / auto_levels lines.
    # The time_step line ("time_step = 45, time_step_fract_num = ...") does not match the line pattern, so dt is NOT in this id;
    # dt is taken from the DT attribute of the output files actually used.
    secs = re.findall(r"&(physics|noah_mp|dynamics|bdy_control)(.*?)\n\s*/", nl, re.S)
    extra = re.findall(r"^\s*(time_step = \d+|e_vert.*|p_top_requested.*|auto_levels_opt.*)$", nl, re.M)
    return hashlib.md5(("".join(a + b for a, b in secs) + "\n".join(extra)).encode()).hexdigest()[:8]
print("| run | segments used (job: partition, tasks, from) | restart times joined | d02 hourly missing | d01 3-hourly missing | duplicated d02 / d01 (bit-identical) | dt d01 / d02 (s) | namelist physics id | output attribute fingerprint d01 / d02 |")
print("|---|---|---|---|---|---|---|---|---|")
for run in RUNS:
    R = [r for r in rows if r["run"] == run]; seen = collections.OrderedDict()
    for r in sorted(R, key=lambda r: r["first"]): seen.setdefault((r["job"], r["start"]), r)
    segtxt = []; rst = []
    for (job, start), r in seen.items():
        q = sacct(job) if job != "?" else dict(part="?", cpus="?")
        frm = "cold" if start == run_start[run] else start
        segtxt.append(f"{job}: {q['part'].replace('participation-cpu', 'participation')}, {q['cpus']}, {frm}")
        if frm != "cold": rst.append(start)
    ck = {c["dom"]: c for c in checks if c["run"] == run}
    dts = lambda d: "/".join(sorted({f"{r['dt']:g}" for r in R if r["dom"] == d}))
    fps = lambda d: "/".join(sorted({r["fp"] for r in R if r["dom"] == d}))
    nl = open(f"{B}/runs/{run}/namelist.base").read()
    print(f"| {run} | {'; '.join(segtxt)} | {', '.join(rst) or 'none (one cold-start segment)'} | {', '.join(ck['d02']['gaps']) or 'none'} | "
          f"{', '.join(ck['d01']['gaps']) or 'none'} | {ck['d02']['dups']} ({ck['d02']['ident']}) / {ck['d01']['dups']} ({ck['d01']['ident']}) | "
          f"{dts('d01')} / {dts('d02')} | {physid(nl)} | {fps('d01')} / {fps('d02')} |")
print()
# ---- merge consecutive files of the same segment into one row per run x segment
print("| run | job | partition | nodes | tasks | started from | d02 hourly frames used | d01 3-hourly frames used | DT d01 / d02 (s) | segment end |")
print("|---|---|---|---|---|---|---|---|---|---|")
for run in RUNS:
    segs = collections.OrderedDict()
    for r in rows:
        if r["run"] != run: continue
        k = (r["job"], r["start"]); s = segs.setdefault(k, dict(job=r["job"], start=r["start"], seg=r["seg"], d02=[], d01=[], dt={}))
        s[r["dom"]].append((r["first"], r["last"], r["n"])); s["dt"][r["dom"]] = r["dt"]
    for (job, start), s in sorted(segs.items(), key=lambda kv: min(x[0] for x in kv[1]["d02"] + kv[1]["d01"])):
        q = sacct(job) if job != "?" else dict(part="?", nodes="?", cpus="?", state="?")
        span = lambda L: (f"{min(x[0] for x in L)} → {max(x[1] for x in L)} ({sum(x[2] for x in L)})" if L else "—")
        sg = s["seg"]; end = ("complete" if sg and sg["ok"] else f"rc {sg['rc']}" if sg and sg["rc"] else f"job {q['state'].lower()}") if sg else "?"
        frm = "cold" if start == run_start[run] else f"restart {start}"
        d1, d2 = s["dt"].get("d01"), s["dt"].get("d02")
        dts = f"{d1:g} / {d2:g}" if d1 is not None and d2 is not None else str(s["dt"])
        print(f"| {run} | {job} | {q['part']} | {q['nodes']} | {q['cpus']} | {frm} | {span(s['d02'])} | {span(s['d01'])} | {dts} | {end} |")
print("\n**Checks of the frames used** (gaps = missing output times; duplicated = times written by two segments, compared variable by variable):\n")
print("| run | domain | frames | first → last | missing | duplicated (bit-identical) | non-finite values |")
print("|---|---|---|---|---|---|---|")
for c in checks:
    print(f"| {c['run']} | {c['dom']} | {c['n']} | {c['first']} → {c['last']} | {', '.join(c['gaps']) or 'none'} | {c['dups']} ({c['ident']}) | {c['nonfinite']} |")
print(f"\nGlobal-attribute fingerprint (physics/dynamics/levels; see the header) of every file used: d01 {sorted(FP.get('d01', []))}, d02 {sorted(FP.get('d02', []))}.")
print("\n**Jobs whose output is not used** (stalled, cancelled, failed at start, retried or superseded):\n")
for run in RUNS:
    L = unused.get(run, [])
    if not L: continue
    parts = []
    for s in sorted(L, key=lambda s: s["t0"]):
        why = "complete" if s["ok"] else (f"rc {s['rc']}" if s["rc"] else f"job {sacct(s['job'])['state'].lower()}")
        sup = ", superseded dt36_mixed version" if run == "ORE_1996_ctl" and s["t0"] < ORE_RERUN else ""
        parts.append(f"{s['job']} ({s['mode']}, from {s['frm']}, {why}{', dt-36 retry' if s['retry36'] else ''}{sup})")
    print(f"- {run}: " + "; ".join(parts))
