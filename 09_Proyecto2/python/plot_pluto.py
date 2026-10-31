#!/usr/bin/env python3
"""
plot_pluto.py - Mapas 2D del MHD Rotor (PLUTO, conf. #01) con pyPLUTO.
Lee ../pluto_run_400/data.000N.dbl y escribe PNGs en ../figuras/.
Uso:  python plot_pluto.py            (snapshots 0-3)
      python plot_pluto.py 3          (solo el snapshot 3)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import pyPLUTO as pp

GAMMA = 1.4
TRANSPOSE = False         # pyPLUTO.display transpone internamente: se le pasa [x, y]
HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "pluto_run_400"
OUT = HERE.parent / "figuras"
OUT.mkdir(exist_ok=True)


def orient_check(pcm, v):
    """Verifica que el array realmente dibujado tenga filas=y, columnas=x."""
    if abs(v[150, 250] - v[250, 150]) < 1e-6:
        return                                   # punto no discriminante
    a = np.asarray(np.ma.getdata(pcm.get_array()))
    if a.ndim == 1 and a.size == v.size:
        a = a.reshape(v.shape)
    if a.shape != v.shape:
        print("  [aviso] no se pudo verificar orientacion, forma", a.shape)
        return
    if not np.isclose(a[250, 150], v[150, 250]):
        raise RuntimeError("Orientacion incorrecta: cambia TRANSPOSE en el script.")


def snapshot(n):
    D = pp.Load(path=str(RUN) + "/", datatype="dbl", nout=n)
    t = float(np.loadtxt(RUN / "dbl.out", usecols=1, ndmin=1)[n])
    x, y = np.array(D.x1), np.array(D.x2)
    f = {k: np.array(v) for k, v in D.d_vars.items()}
    dx, dy = x[1] - x[0], y[1] - y[0]

    # Derivadas
    cs = np.sqrt(GAMMA * f["prs"] / f["rho"])
    vmod = np.sqrt(f["vx1"]**2 + f["vx2"]**2 + f["vx3"]**2)
    mach = vmod / cs
    bmag = np.sqrt(f["Bx1"]**2 + f["Bx2"]**2 + f["Bx3"]**2)
    divB = (f["Bx1s"][1:, :] - f["Bx1s"][:-1, :]) / dx \
         + (f["Bx2s"][:, 1:] - f["Bx2s"][:, :-1]) / dy

    # Diagnosticos para el reporte
    print(f"--- snapshot {n}  t = {t:.5f}")
    print(f"  rho   [{f['rho'].min():.4f}, {f['rho'].max():.4f}]   "
          f"prs [{f['prs'].min():.4f}, {f['prs'].max():.4f}]")
    print(f"  Mach max = {mach.max():.3f}   masa total = {f['rho'].sum()*dx*dy:.6f}")
    print(f"  max|vx3| = {abs(f['vx3']).max():.2e}   max|Bx3| = {abs(f['Bx3']).max():.2e}")
    print(f"  max|divB| (caras CT) = {abs(divB).max():.2e}")

    np.savez(RUN / f"fields_{n:04d}.npz", t=t, x=x, y=y, mach=mach, **f)

    # Paneles: (arreglo, colormap, etiqueta, simetrico)
    P = [(np.log10(f["rho"]), "inferno", r"$\log_{10}\rho$", False),
         (f["prs"],           "viridis", r"$p$",              False),
         (mach,               "magma",   r"$M=|v|/c_s$",      False),
         (bmag,               "plasma",  r"$|B|$",            False),
         (f["vx1"],           "RdBu_r",  r"$v_x$",            True),
         (f["vx2"],           "RdBu_r",  r"$v_y$",            True),
         (f["Bx1"],           "RdBu_r",  r"$B_x$",            True),
         (f["Bx2"],           "RdBu_r",  r"$B_y$",            True)]

    I = pp.Image(LaTeX=False)
    axs = list(np.ravel(I.create_axes(ncol=4, nrow=2, figsize=[26, 12])))
    for k, (arr, cmap, lab, sym) in enumerate(P):
        if sym:
            m = max(abs(arr).max(), 1e-2)
            vmin, vmax = -m, m
        else:
            vmin, vmax = arr.min(), arr.max()
            if vmax - vmin < 1e-6 * max(1.0, abs(vmax)):   # campo casi uniforme
                c = 0.5 * (vmin + vmax); w = 0.1 * max(abs(c), 1.0)
                vmin, vmax = c - w, c + w
        pcm = I.display(arr.T if TRANSPOSE else arr, x1=x, x2=y, ax=axs[k],
                        cmap=cmap, vmin=vmin, vmax=vmax, cpos="right",
                        clabel=lab, title=f"{lab}   t = {t:.3f}",
                        xtitle="x", ytitle="y", aspect="equal")
        orient_check(pcm, arr)
        if k == 0:   # lineas de campo sobre la densidad
            axs[k].streamplot(x, y, f["Bx1"].T, f["Bx2"].T, color="w",
                              linewidth=0.5, density=1.5, arrowsize=0.6)
    out = OUT / f"pluto_rotor_{n:04d}.png"
    I.savefig(str(out), dpi=150)
    print("  figura ->", out)


if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3]
    for n in ns:
        snapshot(n)
