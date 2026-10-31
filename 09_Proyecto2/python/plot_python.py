#!/usr/bin/env python3
"""
plot_python.py - Mapas 2D de la solucion del solver Python (mismos 8 paneles que plot_pluto.py).

Uso:  python plot_python.py                    (N=400, snapshots 0-3)
      python plot_python.py -n 400 --snap 3
Si existe ../pluto_run_400/fields_000k.npz se usan SUS limites de color, para que las
figuras de ambos codigos sean directamente comparables (opcion --no-ref lo desactiva).
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def load(path):
    d = dict(np.load(path))
    d["bmag"] = np.sqrt(d["Bx1"]**2 + d["Bx2"]**2 + d["Bx3"]**2)
    return d


def panels(f):
    return [("log10rho", np.log10(f["rho"]), "inferno", r"$\log_{10}\rho$", False),
            ("prs",      f["prs"],           "viridis", r"$p$",              False),
            ("mach",     f["mach"],          "magma",   r"$M=|v|/c_s$",      False),
            ("bmag",     f["bmag"],          "plasma",  r"$|B|$",            False),
            ("vx1",      f["vx1"],           "RdBu_r",  r"$v_x$",            True),
            ("vx2",      f["vx2"],           "RdBu_r",  r"$v_y$",            True),
            ("Bx1",      f["Bx1"],           "RdBu_r",  r"$B_x$",            True),
            ("Bx2",      f["Bx2"],           "RdBu_r",  r"$B_y$",            True)]


def limits(arr, sym):
    """Mismo criterio que plot_pluto.py."""
    if sym:
        m = max(np.abs(arr).max(), 1e-2)
        return -m, m
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-6 * max(1.0, abs(vmax)):
        c = 0.5 * (vmin + vmax); w = 0.1 * max(abs(c), 1.0)
        return c - w, c + w
    return vmin, vmax


def plot_snapshot(f, out, ref=None, title_extra=""):
    x, y, t = f["x"], f["y"], float(f["t"])
    P = panels(f)
    Pr = panels(ref) if ref is not None else P
    fig, axs = plt.subplots(2, 4, figsize=(26, 12))
    for ax, (key, arr, cmap, lab, sym), (_, arr_r, _, _, _) in zip(axs.ravel(), P, Pr):
        vmin, vmax = limits(arr_r, sym)
        pcm = ax.pcolormesh(x, y, arr.T, cmap=cmap, vmin=vmin, vmax=vmax,
                            shading="nearest", rasterized=True)
        cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(lab, fontsize=14)
        ax.set_title(f"{lab}   t = {t:.3f}{title_extra}", fontsize=15)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        if key == "log10rho":
            ax.streamplot(x, y, f["Bx1"].T, f["Bx2"].T, color="w",
                          linewidth=0.5, density=1.5, arrowsize=0.6)
            ax.set_xlim(x[0] - 0.5 * (x[1] - x[0]), x[-1] + 0.5 * (x[1] - x[0]))
            ax.set_ylim(y[0] - 0.5 * (y[1] - y[0]), y[-1] + 0.5 * (y[1] - y[0]))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("  figura ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--nx", type=int, default=400)
    ap.add_argument("--snap", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--pluto", default=str(HERE.parent / "pluto_run_400"))
    ap.add_argument("--no-ref", action="store_true")
    args = ap.parse_args()
    rundir = HERE.parent / f"python_run_N{args.nx}"
    figdir = HERE.parent / "figuras"
    figdir.mkdir(exist_ok=True)
    for n in args.snap:
        f = load(rundir / f"fields_{n:04d}.npz")
        refp = Path(args.pluto) / f"fields_{n:04d}.npz"
        ref = load(refp) if (refp.exists() and not args.no_ref) else None
        print(f"snapshot {n}  t = {float(f['t']):.5f}" + ("  (limites de PLUTO)" if ref else ""))
        plot_snapshot(f, figdir / f"python_rotor_N{args.nx}_{n:04d}.png", ref)


if __name__ == "__main__":
    main()
