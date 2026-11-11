"""
Makes the paper figures that can be made from your result files (PNG at 300 dpi and PDF).

    python make_figures.py           (needs numpy, pandas, matplotlib; run in the folder with the results)

Inputs : results_eur2_cases.csv, param_refs_eur.npz, reference_solver.py, speed_summary.txt (for Figure 6)
Outputs: fig1_scheme, fig2_error_by_sigma, fig5_ablations, fig6_speed     (.png and .pdf)
Figure 3 comes from plot_examples.py (needs the saved weights) and Figure 4 from learning_curves.py.
Look at every figure before using it: axes, labels, legends and colours are yours to adjust at the top of this file.
"""
import re
import csv
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from reference_solver import european_put

# ---- style (colour-blind-safe palette, Okabe-Ito) ----
C = {"plain": "#6E6E6E", "cheb": "#E69F00", "fourier": "#0072B2", "euro": "#D55E00", "floor": "#009E73"}
LABEL = {"plain": "plain", "cheb": "Chebyshev", "fourier": "Fourier"}
SHORT = {"plain": "plain", "cheb": "Cheb.", "fourier": "Four."}
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 120})
K = 100.0
FLOOR = 0.0157          # mean ATM floor of the penalty method at rho = 30 (from analyze_eur.py, section 2)


def save(fig, name):
    fig.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- data loading (same logic as analyze_eur.py)
NEW_HEADER = ["mode", "N", "rho", "seed", "iters", "domain", "stretch", "sigma", "r", "q", "tau",
              "rel_pen", "max_pen", "rel_ex", "max_ex", "price_err", "price_ex"]


def read_mixed(path):
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        rows = []
        for r in rd:
            if len(r) == len(header):
                rows.append(dict(zip(header, r)))
            elif len(r) == len(NEW_HEADER):
                rows.append(dict(zip(NEW_HEADER, r)))
    d = pd.DataFrame(rows)
    for c in d.columns:
        if c not in ("mode", "domain"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    for c, v in (("domain", "band"), ("stretch", 0)):
        if c not in d.columns:
            d[c] = v
        d[c] = d[c].fillna(v)
    d["variant"] = [b + ("" if dm == "band" else f"_{dm}")
                    + ("" if int(st) == 0 else {1: "_st1", 2: "_lows", 3: "_st1_lows", 4: "_direct"}.get(int(st), f"_st{int(st)}"))
                    + ("" if int(n) in (0, 8) else f"_N{int(n)}") + ("" if int(it) == 30000 else f"_it{int(it)}")
                    for b, dm, st, n, it in zip(d["mode"], d["domain"], d["stretch"], d["N"], d["iters"])]
    return d


def run_means(df, variant, rho=30.0):
    """one number per run (seed): mean ATM error over the cases"""
    s = df[(df.variant == variant) & (df.rho == rho)]
    return s.groupby("seed").price_err.mean().to_numpy()


# ---------------------------------------------------------------- Figure 1: scheme
def fig1():
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

    def box(x, y, w, h, text, fc="#F2F2F2", ec="#444444", fs=7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=1, color="#444444"))

    y0, h0 = 25, 13
    box(1, y0, 15, h0, "Inputs\nx = S/K, τ,\nln σ, r, r − q")
    box(20, y0, 17, h0, "Encoding\nplain | Chebyshev\n| Fourier", fc="#DCEBF7")
    box(41, y0, 14, h0, "MLP\n4 × 96 tanh")
    box(59, y0, 11, h0, "premium\nw")
    box(74, y0, 13, h0, "v = v_E\n+ U·w\n(U = 0.01)")
    box(91, y0, 8, h0, "clip\nmax(v, g)", fs=6.5)
    for x1, x2 in ((16, 20), (37, 41), (55, 59), (70, 74), (87, 91)):
        arrow(x1 + 0.5, y0 + h0 / 2, x2 - 0.5, y0 + h0 / 2)
    box(56, 3, 43, 12, "European price v_E\n(exact Garman–Kohlhagen formula, not learned)", fc="#E8F4EA")
    arrow(80.5, 15.6, 80.5, y0 - 0.6)
    box(1, 3, 49, 12, "Loss (no option data)\nPDE residual + penalty ρ·max(g − v, 0)\n+ initial and boundary conditions", fc="#FFF3D6")
    arrow(64.5, y0 - 0.6, 40, 15.6)
    ax.text(50, 42, "Parametric PINN for American FX puts: one network for all (σ, r, q)", ha="center", fontsize=9, fontweight="bold")
    save(fig, "fig1_scheme")


# ---------------------------------------------------------------- Figure 2: error by sigma
def fig2(df, refs):
    sig = sorted(df.sigma.unique())
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for m in ("plain", "cheb", "fourier"):
        v = f"{m}_rect_it60000"
        s = df[(df.variant == v) & (df.rho == 30.0)]
        if len(s) == 0:
            continue
        per = s.groupby(["seed", "sigma"]).price_err.mean().unstack("seed")
        ax.errorbar(per.index, per.mean(axis=1), yerr=per.std(axis=1), marker="o", ms=4, lw=1.2, capsize=2,
                    color=C[m], label=LABEL[m] + " (60,000 steps)")
    eu = []
    sub = df[(df.variant == "fourier_rect_it60000")]
    for sg in sig:
        e = []
        for _, r in sub[sub.sigma == sg][["r", "q", "tau"]].drop_duplicates().iterrows():
            i = int(np.argmin(abs(refs["sigmas"] - sg))); j = int(np.argmin(abs(refs["rates"] - r.r)))
            l = int(np.argmin(abs(refs["qs"] - r.q))); k = int(np.argmin(abs(refs["taus"] - r.tau)))
            e.append(abs(float(european_put(100, K, r.r, sg, r.tau, r.q)) - refs["v_ex"][i, j, l, k][50] * K))
        eu.append(np.mean(e))
    ax.plot(sig, eu, "--", color=C["euro"], label="European formula (no network)")
    ax.axhline(FLOOR, color=C["floor"], ls=":", label="penalty floor (ρ = 30)")
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xticks(sig); ax.set_xticklabels([f"{s:g}" for s in sig])
    ax.set_xlabel("volatility σ"); ax.set_ylabel("mean ATM error (price units, strike = 100)")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2); ax.grid(alpha=0.25, which="both")
    save(fig, "fig2_error_by_sigma")


# ---------------------------------------------------------------- Figure 5: ablations
run_means_cache = {}


def points(ax, items, title):
    """items: list of (variant, colour key, tick label or None for a gap). Markers = mean over seeds, whiskers = sd."""
    x, ticks, labels = 0, [], []
    for v, ck, lab in items:
        if v is None:
            x += 0.8
            continue
        vals = run_means_cache.get(v, np.array([]))
        if len(vals):
            ax.errorbar(x, vals.mean(), yerr=vals.std(ddof=1) if len(vals) > 1 else 0, fmt="o", ms=5, color=C[ck], capsize=3, lw=1.3)
            ax.plot([x] * len(vals), vals, ".", color=C[ck], alpha=0.35, ms=3)      # the individual seeds
        ticks.append(x); labels.append(lab)
        x += 1
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7)
    ax.set_yscale("log"); ax.set_title(title, fontsize=9); ax.grid(alpha=0.25, axis="y", which="both")
    ax.axhline(0.1235, color=C["euro"], ls="--", lw=0.8)


def fig5(df):
    for v in set(df.variant):
        run_means_cache[v] = run_means(df, v)
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.2), sharey=True, gridspec_kw={"width_ratios": [1.5, 1.0, 1.5]})
    A = []
    for m in ("plain", "cheb", "fourier"):
        A += [(m, m, f"{SHORT[m]}\nband"), (f"{m}_rect", m, f"{SHORT[m]}\nrect"), (None, None, None)]
    points(axes[0], A[:-1], "(A) training domain")
    points(axes[1], [("fourier_rect_N4", "fourier", "N = 4"), ("fourier_rect", "fourier", "N = 8"),
                     ("fourier_rect_N12", "fourier", "N = 12"), ("fourier_rect_N16", "fourier", "N = 16")], "(B) Fourier features")
    Cc = []
    for m in ("plain", "cheb", "fourier"):
        Cc += [(f"{m}_rect", m, f"{SHORT[m]}\npremium"), (f"{m}_rect_direct", m, f"{SHORT[m]}\ndirect"), (None, None, None)]
    points(axes[2], Cc[:-1], "(C) premium vs direct")
    axes[0].set_ylabel("mean ATM error (price units)")
    ticks = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    axes[0].set_yticks(ticks); axes[0].set_yticklabels([f"{t:g}" for t in ticks]); axes[0].set_ylim(0.015, 1.3)
    for a in axes:
        a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[2].text(0.98, 0.1235 * 1.06, "European formula", fontsize=7, color=C["euro"], ha="right", transform=axes[2].get_yaxis_transform())
    for a in axes:
        a.tick_params(axis="x", labelsize=6.5)
    fig.text(0.5, -0.03, "30,000 training steps. Large marker = mean over seeds, whisker = standard deviation, dots = single seeds (3-5 seeds).",
             ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "fig5_ablations")


# ---------------------------------------------------------------- Figure 6: speed (from speed_summary.txt)
def fig6(path="speed_summary.txt"):
    txt = open(path).read()
    batch = re.findall(r"B=\s*(\d+):\s*([\d.]+) ms per call\s+->\s+([\d.]+) microseconds per contract", txt)
    items = []
    if batch:
        b = batch[-1]
        items.append((f"PINN, batch of {b[0]}", float(b[2]) / 1000.0, None))
    # table lines look like:  "  <method name>      <ms per contract>   <error>"; the method name may contain digits and brackets
    for line in txt.splitlines():
        m = re.match(r"^\s{2}(PINN \(single.*?|FD penalized.*?|PSOR.*?)\s+([\d.]+)\s+([\d.]+)\s*$", line)
        if m:
            items.append((m.group(1).strip(), float(m.group(2)), float(m.group(3))))
    if len(items) < 3:
        print("WARNING: could not parse the finite-difference rows of", path)
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    names = [i[0] for i in items][::-1]; vals = [i[1] for i in items][::-1]; errs = [i[2] for i in items][::-1]
    cols = [C["fourier"] if n.startswith("PINN") else C["plain"] for n in names]
    ax.barh(range(len(names)), vals, color=cols)
    ax.set_yticks(range(len(names))); ax.set_yticklabels([n.replace(" (M=", "\n(M=") for n in names], fontsize=7)
    for y, (v, e) in enumerate(zip(vals, errs)):
        ax.text(v * 1.15, y, f"{v:g} ms" + (f", error {e:.4f}" if e else ""), va="center", fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel("time per contract (ms, log scale)"); ax.set_xlim(0.1, max(vals) * 25)
    ax.grid(alpha=0.25, axis="x", which="both")
    save(fig, "fig6_speed")


if __name__ == "__main__":
    fig1()
    if os.path.exists("results_eur2_cases.csv") and os.path.exists("param_refs_eur.npz"):
        data = read_mixed("results_eur2_cases.csv")
        refs = np.load("param_refs_eur.npz")
        fig2(data, refs)
        fig5(data)
    else:
        print("results_eur2_cases.csv / param_refs_eur.npz not found: skipped figures 2 and 5")
    if os.path.exists("speed_summary.txt"):
        fig6()
    else:
        print("speed_summary.txt not found: skipped figure 6")
