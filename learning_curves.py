"""
Learning curves from your training logs (error versus training step), for the paper's convergence figure.

    python learning_curves.py            (needs matplotlib, numpy; reads logs2/ logs4/ logs5/ logs6/ in the current folder)

Uses the default configuration (N=8, no stretch / low-sigma emphasis) in the rect domain, rho given by the log, and compares plain /
cheb / fourier at equal training length. y = mean ATM error over the 105 test cases (price units), averaged over the seeds found.
Writes learning_curves.png and learning_curves.txt (small; send me the txt).
"""
import re
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(r"^it\s+(\d+) \| rho\s+([\d.]+) \| loss [\d.eE+-]+ \| mean rel L2 vs penalized [\d.eE+-]+ \| mean price err S=100 ([\d.]+) \| worst ([\d.]+) \| (\d+)s")
HEAD = re.compile(r"^MODE=(\w+) N=(\S+) SEED=(\d+) \| inputs \d+ \| parameters \d+ \| steps (\d+)")
runs = {}
for f in sorted(glob.glob("logs*/*.log")):
    name = os.path.basename(f)
    if name.startswith(("smoke", "val_eval", "choice")):
        continue
    text = open(f).read().splitlines()
    if not any(l.startswith("FINAL") for l in text):
        continue
    head = next((HEAD.match(l) for l in text if HEAD.match(l)), None)
    pts = [LINE.match(l) for l in text if LINE.match(l)]
    if head is None or not pts:
        continue
    mode, N, seed, steps = head.group(1), head.group(2), int(head.group(3)), int(head.group(4))
    if "_N" in name and "_N8_" not in name:
        continue                                              # skip feature-count ablations
    if re.search(r"_st[123]", name):
        continue                                              # skip stretch / low-sigma-emphasis variants
    dom = "rect" if "rect" in name else ("box" if "dombox" in name else ("wide" if "domwide" in name else "band"))
    rho = int(round(float(pts[-1].group(2))))
    key = (dom, mode, steps, rho)
    runs.setdefault(key, {})[seed] = (np.array([int(m.group(1)) for m in pts]), np.array([float(m.group(3)) for m in pts]))

def mean_curve(key):
    d = runs[key]
    steps = next(iter(d.values()))[0]
    ys = np.array([v[1] for v in d.values() if len(v[1]) == len(steps)])
    return steps, ys.mean(axis=0), len(ys)

out = []
lengths = [n for n in (30000, 60000, 120000) if any(k[0] == "rect" and k[2] == n for k in runs)]
fig, axes = plt.subplots(1, len(lengths), figsize=(6 * len(lengths), 4.5), sharey=True, squeeze=False)
axes = axes[0]
colors = {"plain": "tab:gray", "cheb": "tab:orange", "fourier": "tab:blue"}
for ax, steps_total in zip(axes, lengths):
    for mode in ("plain", "cheb", "fourier"):
        for rho, ls in ((30, "-"), (100, "--")):
            key = ("rect", mode, steps_total, rho)
            if key in runs:
                x, y, n = mean_curve(key)
                ax.plot(x, y, ls, color=colors[mode], label=f"{mode}, rho={rho} ({n} seeds)")
                mid = y[np.argmin(np.abs(x - steps_total / 2))]
                out.append(f"rect, {steps_total} steps, {mode}, rho={rho}, {n} seeds: ATM error at half of training {mid:.4f}, at the end {y[-1]:.4f}")
    ax.axhline(0.1235, color="red", lw=0.8, ls=":", label="European formula (no network)")
    ax.axhline(0.0157, color="green", lw=0.8, ls=":", label="penalty floor, rho=30")
    ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_title(f"{steps_total} training steps (rect domain)"); ax.grid(alpha=.3)
axes[0].set_ylabel("mean ATM error (price units)"); axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("learning_curves.png", dpi=150)
open("learning_curves.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out)); print("Saved learning_curves.png and learning_curves.txt")
