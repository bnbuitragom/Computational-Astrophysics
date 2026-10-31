#!/usr/bin/env python3
"""
compare.py - Comparacion cuantitativa PLUTO vs Python para el MHD Rotor.

Lee fields_000k.npz de ambos codigos (mismo formato), y para cada snapshot:
  * Normas de error por variable (relativas a la norma de PLUTO):
        L1  = sum|a-b| / sum|b|,   L2 = sqrt(sum (a-b)^2) / sqrt(sum b^2),
        Linf/rango = max|a-b| / (max b - min b),  P99/rango = percentil 99 de |a-b| / rango
  * Cortes 1D superpuestos (fila y=+dy/2 y columna x=+dx/2) con residuo
  * Mapas 2D: PLUTO | Python | diferencia, y mapas de diferencia de las 8 variables
  * Error de simetria central (rotacion 180 grados) de cada codigo
  * Conservacion: masa, max|div B|
Tabla de errores -> ../reporte/errores_N{N}.csv ; figuras -> ../figuras/

Uso:  python compare.py                       (N=400)
      python compare.py --pluto ../pluto_run_400 --python ../python_run_N400
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

# (clave, etiqueta, paridad bajo rotacion de 180 grados: +1 par, -1 impar)
VARS = [("rho", r"$\rho$", +1), ("prs", r"$p$", +1), ("vx1", r"$v_x$", -1), ("vx2", r"$v_y$", -1),
        ("Bx1", r"$B_x$", +1), ("Bx2", r"$B_y$", +1), ("mach", r"$M$", +1), ("bmag", r"$|B|$", +1)]


def load(path):
    d = dict(np.load(path))
    d["bmag"] = np.sqrt(d["Bx1"]**2 + d["Bx2"]**2 + d["Bx3"]**2)
    return d


def norms(a, b):
    d = a - b
    rng = b.max() - b.min()
    nb1, nb2 = np.abs(b).sum(), np.sqrt((b**2).sum())
    return dict(L1=np.abs(d).sum() / nb1 if nb1 > 0 else np.nan,
                L2=np.sqrt((d**2).sum()) / nb2 if nb2 > 0 else np.nan,
                Linf_rng=np.abs(d).max() / rng if rng > 0 else np.nan,
                P99_rng=np.percentile(np.abs(d), 99) / rng if rng > 0 else np.nan,
                L1_abs=np.abs(d).mean())


def symmetry_error(f):
    """Error relativo (L1/rango) de la simetria  f(-x,-y) = parity * f(x,y)."""
    out = {}
    for k, _, par in VARS:
        a = f[k]
        rot = a[::-1, ::-1] * par
        rng = a.max() - a.min()
        out[k] = np.abs(a - rot).mean() / rng if rng > 0 else 0.0
    return out


def divB(f):
    dx, dy = f["x"][1] - f["x"][0], f["y"][1] - f["y"][0]
    return np.abs((f["Bx1s"][1:, :] - f["Bx1s"][:-1, :]) / dx
                  + (f["Bx2s"][:, 1:] - f["Bx2s"][:, :-1]) / dy).max()


def plot_cuts(fp, fy, n, outdir, tag):
    x, y, t = fp["x"], fp["y"], float(fp["t"])
    N = x.size
    for cut, label in (("y", f"y = {y[N//2]:+.5f}"), ("x", f"x = {x[N//2]:+.5f}")):
        fig = plt.figure(figsize=(22, 11))
        gs = fig.add_gridspec(4, 4, height_ratios=[3, 1, 3, 1], hspace=0.75, wspace=0.28)
        for k, (key, lab, _) in enumerate(VARS):
            r, c = divmod(k, 4)
            a1 = fig.add_subplot(gs[2 * r, c]); a2 = fig.add_subplot(gs[2 * r + 1, c], sharex=a1)
            s = (slice(None), N // 2) if cut == "y" else (N // 2, slice(None))
            coord = x if cut == "y" else y
            p, q = fp[key][s], fy[key][s]
            a1.plot(coord, p, "k-", lw=1.6, label="PLUTO")
            a1.plot(coord, q, "r--", lw=1.1, label="Python")
            a1.set_title(f"{lab}   ({label}, t = {t:.3f})", fontsize=12)
            a2.plot(coord, q - p, "b-", lw=0.9)
            a2.axhline(0, color="gray", lw=0.5)
            a2.set_ylabel("Py - PLUTO", fontsize=8)
            a2.set_xlabel("x" if cut == "y" else "y")
            plt.setp(a1.get_xticklabels(), visible=False)
            if k == 0:
                a1.legend()
        fig.suptitle(f"Corte a lo largo de {'x' if cut == 'y' else 'y'}   {tag}", fontsize=15)
        out = outdir / f"compare_cut_{cut}_{n:04d}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("  figura ->", out)


def plot_maps(fp, fy, n, outdir, tag):
    x, y, t = fp["x"], fp["y"], float(fp["t"])
    # (1) PLUTO | Python | diferencia para rho y p
    fig, axs = plt.subplots(2, 3, figsize=(18, 11.5))
    for r, (key, lab, cm, tr) in enumerate([("rho", r"$\log_{10}\rho$", "inferno", np.log10),
                                            ("prs", r"$p$", "viridis", lambda a: a)]):
        A, B = tr(fp[key]), tr(fy[key])
        vmin, vmax = A.min(), A.max()
        for c, (arr, ttl) in enumerate([(A, "PLUTO"), (B, "Python")]):
            pcm = axs[r, c].pcolormesh(x, y, arr.T, cmap=cm, vmin=vmin, vmax=vmax,
                                       shading="nearest", rasterized=True)
            fig.colorbar(pcm, ax=axs[r, c], fraction=0.046, pad=0.03).set_label(lab)
            axs[r, c].set_title(f"{ttl}: {lab}   t = {t:.3f}")
        D = B - A
        m = np.abs(D).max()
        pcm = axs[r, 2].pcolormesh(x, y, D.T, cmap="RdBu_r", vmin=-m, vmax=m,
                                   shading="nearest", rasterized=True)
        fig.colorbar(pcm, ax=axs[r, 2], fraction=0.046, pad=0.03).set_label("Python - PLUTO")
        axs[r, 2].set_title(f"Diferencia ({lab})")
        for c in range(3):
            axs[r, c].set_aspect("equal"); axs[r, c].set_xlabel("x"); axs[r, c].set_ylabel("y")
    fig.suptitle(tag, fontsize=15)
    fig.tight_layout()
    out = outdir / f"compare_maps_rho_p_{n:04d}.png"
    fig.savefig(out, dpi=130); plt.close(fig); print("  figura ->", out)
    # (2) diferencias de las 8 variables, normalizadas por el rango de PLUTO
    fig, axs = plt.subplots(2, 4, figsize=(26, 12))
    for ax, (key, lab, _) in zip(axs.ravel(), VARS):
        rng = fp[key].max() - fp[key].min()
        D = (fy[key] - fp[key]) / (rng if rng > 0 else 1.0)
        m = max(np.abs(D).max(), 1e-12)
        pcm = ax.pcolormesh(x, y, D.T, cmap="RdBu_r", vmin=-m, vmax=m, shading="nearest", rasterized=True)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03).set_label("(Py - PLUTO) / rango PLUTO")
        ax.set_title(f"{lab}   max|dif|/rango = {m:.2e}", fontsize=14)
        ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle(f"Diferencias normalizadas, t = {t:.3f}   {tag}", fontsize=15)
    fig.tight_layout()
    out = outdir / f"compare_diff_all_{n:04d}.png"
    fig.savefig(out, dpi=130); plt.close(fig); print("  figura ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pluto", default=str(HERE.parent / "pluto_run_400"))
    ap.add_argument("--python", default=str(HERE.parent / "python_run_N400"))
    ap.add_argument("--snap", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    pdir, ydir = Path(args.pluto), Path(args.python)
    figdir, repdir = HERE.parent / "figuras", HERE.parent / "reporte"
    figdir.mkdir(exist_ok=True); repdir.mkdir(exist_ok=True)

    rows, N = [], None
    for n in args.snap:
        fp, fy = load(pdir / f"fields_{n:04d}.npz"), load(ydir / f"fields_{n:04d}.npz")
        N = fp["x"].size
        if fp["rho"].shape != fy["rho"].shape:
            raise SystemExit(f"Las mallas no coinciden: {fp['rho'].shape} vs {fy['rho'].shape}")
        dt = abs(float(fp["t"]) - float(fy["t"]))
        assert dt < 1e-8, f"snapshot {n}: los tiempos difieren ({fp['t']} vs {fy['t']})"
        tag = args.tag or f"N = {N}"
        print(f"\n=== snapshot {n}   t = {float(fp['t']):.5f}   (|dt| = {dt:.1e}) ===")
        print(f"{'var':>5} {'L1':>10} {'L2':>10} {'Linf/rng':>10} {'P99/rng':>10} {'L1 abs':>10}")
        for key, _, _ in VARS:
            e = norms(fy[key], fp[key])
            print(f"{key:>5} {e['L1']:10.3e} {e['L2']:10.3e} {e['Linf_rng']:10.3e} "
                  f"{e['P99_rng']:10.3e} {e['L1_abs']:10.3e}")
            rows.append(dict(snapshot=n, t=float(fp["t"]), var=key, **e))
        dxdy = (fp["x"][1] - fp["x"][0]) * (fp["y"][1] - fp["y"][0])
        sp, sy_ = symmetry_error(fp), symmetry_error(fy)
        print(f"  masa:   PLUTO {fp['rho'].sum()*dxdy:.8f}   Python {fy['rho'].sum()*dxdy:.8f}")
        print(f"  max|divB|: PLUTO {divB(fp):.2e}   Python {divB(fy):.2e}")
        print("  asimetria central (L1/rango): " + "  ".join(
            f"{k}: {sp[k]:.1e}/{sy_[k]:.1e}" for k, _, _ in VARS[:4]) + "   [PLUTO/Python]")
        if not args.no_plots:
            plot_cuts(fp, fy, n, figdir, tag)
            plot_maps(fp, fy, n, figdir, tag)

    out = repdir / f"errores_N{N}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nTabla de errores ->", out)


if __name__ == "__main__":
    main()
