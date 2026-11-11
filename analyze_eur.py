"""
Summarizes the EUR/USD experiments.

    python analyze_eur.py [csv_file] [restricted]          (needs: pip install pandas   -> use  pip install "numpy<2" pandas)

    csv_file   default results_eur_cases.csv;  use results_eur2_cases.csv for batches 2-4 and
               results_eur_clip_cases.csv for the re-evaluated first batch
    restricted keep only cases with realistic carry r - q in [-0.01, 0.04]

Prints the tables below and writes them to eur_summary.txt (send me that small file instead of the big CSV).
Each RUN (mode, rho, seed, variant) is the independent unit; case-level numbers are first averaged within a run.
ATM error = |PINN price - exact American price| at S=100, in price units (strike = 100, so 0.1 = 10 bps of strike).
'EUR/USD-weighted' ATM error weights the 7 test volatilities by how often the 21-day realized volatility of your EUR/USD file
fell closest to each of them (3.0%: 5.8%, 4.5%: 18.3%, 6%: 28.3%, 8%: 27.7%, 10%: 15.4%, 12.5%: 2.9%, 15%: 1.5%).
Variant names: base mode (plain/cheb/fourier) + _box/_rect/_wide (training domain) + _st1 (stretched x input) + _lows (low-sigma emphasis) + _direct (no European + premium split)
               + _N<n> (feature count if not 8) + _it<steps> (if not 30000 steps). The plain names are the default configuration.
"""
import io
import os
import sys
import numpy as np
import pandas as pd
from reference_solver import european_put

K = 100.0
W = {0.03: 0.058, 0.045: 0.183, 0.06: 0.283, 0.08: 0.277, 0.1: 0.154, 0.125: 0.029, 0.15: 0.015}
path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "restricted" else "results_eur_cases.csv"
restricted = "restricted" in sys.argv[1:]


def add_variant(d):
    d = d.copy()
    for c, v in (("domain", "band"), ("stretch", 0)):
        if c not in d.columns:
            d[c] = v
        d[c] = d[c].fillna(v)
    d["base_mode"] = d["mode"]
    d["variant"] = [b + ("" if dm == "band" else f"_{dm}") + ("" if int(st) == 0 else {1: "_st1", 2: "_lows", 3: "_st1_lows", 4: "_direct"}.get(int(st), f"_st{int(st)}")) + ("" if int(n) in (0, 8) else f"_N{int(n)}")
                    + ("" if int(it) == 30000 else f"_it{int(it)}")
                    for b, dm, st, n, it in zip(d.base_mode, d.domain, d.stretch, d.N, d.iters)]
    return d



NEW_CASES_HEADER = ["mode", "N", "rho", "seed", "iters", "domain", "stretch", "sigma", "r", "q", "tau",
                    "rel_pen", "max_pen", "rel_ex", "max_ex", "price_err", "price_ex"]


def read_mixed(path):
    """Reads a results CSV whose rows may follow two layouts: the header's own (older runs) and the 17-column layout
    written by the later runs (extra 'domain' and 'stretch' columns that were appended below the old header)."""
    import csv
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        rows, skipped = [], 0
        for r in rd:
            if len(r) == len(header):
                rows.append(dict(zip(header, r)))
            elif len(r) == len(NEW_CASES_HEADER):
                rows.append(dict(zip(NEW_CASES_HEADER, r)))
            else:
                skipped += 1
    d = pd.DataFrame(rows)
    for c in d.columns:
        if c not in ("mode", "domain"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if skipped:
        print(f"WARNING: skipped {skipped} rows with an unexpected number of fields")
    return d


df = add_variant(read_mixed(path))
df["mode"] = df["variant"]
df["carry"] = (df.r - df.q).round(4)
if restricted:
    df = df[(df.carry >= -0.01 - 1e-9) & (df.carry <= 0.04 + 1e-9)].copy()
REF = np.load("param_refs_eur.npz")
idx = lambda arr, v: int(np.argmin(np.abs(arr - v)))

eu = []
for s, r, q, t in zip(df.sigma, df.r, df.q, df.tau):
    i, j, l, k = idx(REF["sigmas"], s), idx(REF["rates"], r), idx(REF["qs"], q), idx(REF["taus"], t)
    eu.append(abs(float(european_put(100, K, r, s, t, q)) - REF["v_ex"][i, j, l, k][50] * K))
df["euro_err"] = eu
df["better_than_euro"] = df.price_err < df.euro_err

out = io.StringIO()
def p(*a):
    print(*a); print(*a, file=out)
def show(title, table):
    p("\n" + title); p(table.to_string())


def wmean(g, col="price_err"):
    m = g.groupby(g.sigma.round(3))[col].mean()
    keep = [k for k in m.index if k in W]
    w = pd.Series({k: W[k] for k in keep})
    return float((m.loc[keep] * w).sum() / w.sum())


keys = ["rho", "mode", "base_mode", "domain", "stretch", "iters", "seed"]
ncase = df.groupby(["mode", "rho", "seed"]).size().iloc[0]
p(f"File: {path}{' (restricted carry)' if restricted else ''} | runs: {df.groupby(['mode', 'rho', 'seed']).ngroups} | cases per run: {ncase}")
p(df.groupby(["rho", "mode"]).seed.nunique().unstack().rename_axis(index="rho").T.to_string())

run = df.groupby(keys).agg(atm_mean=("price_err", "mean"), atm_median=("price_err", "median"), atm_worst=("price_err", "max"),
        rel_pen=("rel_pen", "mean"), rel_ex=("rel_ex", "mean"), euro_mean=("euro_err", "mean"), beat=("better_than_euro", "mean")).reset_index()
run["atm_weighted"] = df.groupby(keys).apply(wmean, include_groups=False).values if "include_groups" in pd.DataFrame.groupby.__code__.co_varnames else df.groupby(keys).apply(wmean).values
run["rel_pen"] *= 100; run["rel_ex"] *= 100
agg = run.groupby(["rho", "mode"]).agg(n_seeds=("seed", "count"), atm_mean=("atm_mean", "mean"), atm_sd=("atm_mean", "std"),
        atm_weighted=("atm_weighted", "mean"), atm_median=("atm_median", "mean"), atm_worst=("atm_worst", "mean"),
        rel_L2_vs_pen_pct=("rel_pen", "mean"), rel_L2_vs_exact_pct=("rel_ex", "mean"), share_better_than_euro=("beat", "mean"))
show("1) OVERALL per rho and variant (mean over seeds of the per-run means; atm_weighted = EUR/USD-weighted ATM error)", agg.round(4))
first = df.groupby(["mode", "rho", "seed"]).ngroup().min()
ref_run = df[(df["mode"] == df["mode"].iloc[0]) & (df.seed == df.seed.iloc[0]) & (df.rho == df.rho.iloc[0])]
p(f"\nEuropean-formula benchmark (no network), same cases: mean ATM error {ref_run.euro_err.mean():.4f}; EUR/USD-weighted {wmean(ref_run, 'euro_err'):.4f}")

keys_c = df[["sigma", "r", "q", "tau"]].drop_duplicates().to_numpy()
ii = [(idx(REF["sigmas"], a), idx(REF["rates"], b), idx(REF["qs"], c), idx(REF["taus"], d_)) for a, b, c, d_ in keys_c]
g0 = np.maximum(1 - REF["xs"], 0)
fl = []
for rho in sorted(df.rho.unique()):
    vp = np.array([REF[f"v_pen_{int(rho)}"][c] for c in ii]); ve = np.array([REF["v_ex"][c] for c in ii]); vc = np.maximum(vp, g0)
    for name, v in (("raw", vp), ("clipped", vc)):
        gap = np.abs(v - ve)
        fl.append(dict(rho=int(rho), version=name, floor_ATM_err=gap[:, 50].mean() * K,
                       floor_rel_L2_vs_exact_pct=np.mean(np.linalg.norm(gap, axis=-1) / np.linalg.norm(ve, axis=-1)) * 100))
show("2) FLOOR set by the penalty method itself (score of a perfect network vs the exact answer)", pd.DataFrame(fl).set_index(["rho", "version"]).round(4))

for rho in sorted(df.rho.unique()):
    sub = df[df.rho == rho]
    t1 = sub.pivot_table(index="sigma", columns="mode", values="price_err", aggfunc="mean")
    t1["European"] = sub[sub["mode"] == sub["mode"].iloc[0]].groupby("sigma").euro_err.mean()
    show(f"3) rho={int(rho)}: mean ATM error by sigma", t1.round(4))
    show(f"   rho={int(rho)}: mean ATM error by carry r - q", sub.pivot_table(index="carry", columns="mode", values="price_err", aggfunc="mean").round(4))

if df.rho.nunique() > 1:
    show("4) Effect of rho on the mean ATM error (seed 0)", run[run.seed == 0].pivot_table(index="mode", columns="rho", values="atm_mean").round(4))

rows = []
dflt = run[~run["mode"].str.contains("_N")]
for (rho, dom, st, itr), grp in dflt.groupby(["rho", "domain", "stretch", "iters"]):
    for metric in ("atm_mean", "atm_weighted"):
        w = grp.pivot_table(index="seed", columns="base_mode", values=metric).dropna()
        if len(w) >= 1 and {"plain", "cheb", "fourier"} <= set(w.columns):
            rows.append(dict(rho=int(rho), domain=dom, stretch=int(st), iters=int(itr), metric=metric, seeds=len(w),
                             cheb_over_plain=float(np.exp(np.log(w.cheb / w.plain).mean())),
                             fourier_over_plain=float(np.exp(np.log(w.fourier / w.plain).mean())),
                             cheb_over_fourier=float(np.exp(np.log(w.cheb / w.fourier).mean())),
                             cheb_beats_plain=f"{int((w.cheb < w.plain).sum())}/{len(w)}",
                             fourier_beats_plain=f"{int((w.fourier < w.plain).sum())}/{len(w)}",
                             fourier_beats_cheb=f"{int((w.fourier < w.cheb).sum())}/{len(w)}"))
if rows:
    show("5) Mode comparison seed by seed per configuration: geometric-mean ratio (below 1 = the first is better). Meaningful with >= 3 seeds.",
         pd.DataFrame(rows).set_index(["rho", "domain", "stretch", "iters", "metric"]).round(3))
pr = dflt[dflt.rho == dflt.rho.min()].pivot_table(index="seed", columns="mode", values="atm_mean")
show(f"   per-seed mean ATM error at rho={int(dflt.rho.min())} (default-N variants)", pr.round(4))

if path.endswith("results_eur2_cases.csv") and os.path.exists("results_val.csv"):
    v = add_variant(pd.read_csv("results_val.csv"))
    tv = v.groupby(["variant", "rho"]).agg(seeds=("seed", "count"), val_atm_mean=("val_atm_mean", "mean"), val_atm_sd=("val_atm_mean", "std"),
                                  val_rel_L2_pct=("val_rel_ex", lambda x: x.mean() * 100), test_atm_mean=("test_atm_mean", "mean"))
    show("7) VALIDATION contracts (60 held-out EUR/USD-like contracts) per variant and rho, with the test-grid ATM error for comparison", tv.round(4))

worst = df[(df.rho == df.rho.min())].sort_values("price_err", ascending=False).head(6)[["mode", "seed", "sigma", "r", "q", "tau", "price_err", "euro_err"]]
show(f"6) Six worst cases at rho={int(df.rho.min())}", worst.round(4).set_index("mode"))
open("eur_summary.txt", "w").write(out.getvalue())
print("\nSaved eur_summary.txt")
