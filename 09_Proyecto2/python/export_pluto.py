#!/usr/bin/env python3
"""
export_pluto.py - Convierte la salida .dbl de PLUTO (cualquier resolucion) a fields_000k.npz
con el MISMO formato que escribe run_rotor.py, para usar compare.py / convergence.py.

Uso:  python export_pluto.py ../pluto_run_400
      python export_pluto.py ../pluto_run_200 --snap 3
El directorio debe contener data.000k.dbl, grid.out y dbl.out (copiados de la corrida de PLUTO).
Los .npz se guardan comprimidos (para que el repositorio pese menos).
"""
import argparse
from pathlib import Path
import numpy as np
import pyPLUTO as pp

GAMMA = 1.4


def export(rundir, snaps=None):
    rundir = Path(rundir).resolve()
    times = np.loadtxt(rundir / "dbl.out", usecols=1, ndmin=1)
    for n in (snaps if snaps else range(len(times))):
        D = pp.Load(path=str(rundir) + "/", datatype="dbl", nout=n)
        f = {k: np.array(v) for k, v in D.d_vars.items()}
        x, y = np.array(D.x1), np.array(D.x2)
        cs = np.sqrt(GAMMA * f["prs"] / f["rho"])
        mach = np.sqrt(f["vx1"]**2 + f["vx2"]**2 + f["vx3"]**2) / cs
        out = rundir / f"fields_{n:04d}.npz"
        np.savez_compressed(out, t=float(times[n]), x=x, y=y, mach=mach, **f)
        print(f"  {out.name}: t = {times[n]:.5f}  N = {x.size}  rho[{f['rho'].min():.4f}, {f['rho'].max():.4f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--snap", type=int, nargs="+", default=None)
    a = ap.parse_args()
    export(a.rundir, a.snap)
