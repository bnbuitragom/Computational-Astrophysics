#!/usr/bin/env python3
"""
run_rotor.py - MHD Rotor (config. #01 de PLUTO) con el solver Python de mhd_core.py.

Condicion inicial IDENTICA a init.c: gamma=1.4, p=1, omega=20, r0=0.1, r1=0.115,
Bx=5/sqrt(4 pi), dominio [-0.5,0.5]^2, fronteras outflow, tstop=0.15, CFL=0.4,
first_dt=1e-6 y crecimiento maximo de dt de 1.1 por paso (CFL_max_var).

Los tiempos de salida por defecto se leen de dbl.out de PLUTO, de modo que los
snapshots de ambos codigos corresponden exactamente al mismo instante.

Uso:  python run_rotor.py                  (N=400, tiempos de ../pluto_run_400/dbl.out)
      python run_rotor.py -n 200           (otra resolucion)
      python run_rotor.py -n 100 --times 0 0.05 0.1 0.15
Salida: ../python_run_N{N}/fields_000k.npz  con las mismas claves que los de PLUTO.
"""
import argparse
import time
from pathlib import Path
import numpy as np
from mhd_core import MHDSolver2D, NG

GAMMA = 1.4
OMEGA, R0, R1 = 20.0, 0.1, 0.115
BX0 = 5.0 / np.sqrt(4.0 * np.pi)
CFL, FIRST_DT, DT_GROWTH = 0.4, 1.0e-6, 1.1


def initial_condition(x, y):
    """Primitivas en centros de celda, igual que Init() en init.c (GEOMETRY == CARTESIAN)."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt(X**2 + Y**2)
    f = (R1 - r) / (R1 - R0)
    V = np.zeros((8, x.size, y.size))
    V[0] = 1.0
    V[7] = 1.0
    V[4] = BX0
    inner = r <= R0
    taper = (r > R0) & (r < R1)
    V[0][inner] = 10.0
    V[1][inner] = -OMEGA * Y[inner]
    V[2][inner] = OMEGA * X[inner]
    V[0][taper] = 1.0 + 9.0 * f[taper]
    V[1][taper] = -f[taper] * OMEGA * Y[taper] * R0 / r[taper]
    V[2][taper] = f[taper] * OMEGA * X[taper] * R0 / r[taper]
    return V


def save_snapshot(s, n, outdir):
    V = s.primitives()
    bxs, bys = s.faces()
    cs = np.sqrt(GAMMA * V[7] / V[0])
    mach = np.sqrt(V[1]**2 + V[2]**2 + V[3]**2) / cs
    np.savez_compressed(outdir / f"fields_{n:04d}.npz", t=s.t, x=s.x, y=s.y, mach=mach,
             rho=V[0], vx1=V[1], vx2=V[2], vx3=V[3], Bx1=V[4], Bx2=V[5], Bx3=V[6],
             prs=V[7], Bx1s=bxs, Bx2s=bys)
    print(f"  [snapshot {n}] t={s.t:.5f}  rho[{V[0].min():.4f},{V[0].max():.4f}] "
          f"prs[{V[7].min():.4f},{V[7].max():.4f}]  Mach max={mach.max():.3f}  "
          f"masa={s.mass():.6f}  max|divB|={s.max_divB():.2e}", flush=True)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--nx", type=int, default=400, help="celdas por lado")
    ap.add_argument("--times", type=float, nargs="+", default=None, help="tiempos de salida")
    ap.add_argument("--pluto-dbl", default=str(here.parent / "pluto_run_400" / "dbl.out"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.times is not None:
        times = np.array(args.times)
    elif Path(args.pluto_dbl).exists():
        times = np.loadtxt(args.pluto_dbl, usecols=1, ndmin=1)
        print("tiempos de salida leidos de", args.pluto_dbl, "->", times)
    else:
        times = np.array([0.0, 0.05, 0.10, 0.15])
    outdir = Path(args.out) if args.out else here.parent / f"python_run_N{args.nx}"
    outdir.mkdir(parents=True, exist_ok=True)

    N = args.nx
    s = MHDSolver2D(N, N, (-0.5, 0.5), (-0.5, 0.5), gamma=GAMMA, cfl=CFL, bc="outflow")
    V0 = initial_condition(s.x, s.y)
    bxs = np.full((N + 1, N), BX0)           # A_z = Bx*y  ->  Bx = BX0, By = 0 en las caras
    bys = np.zeros((N, N + 1))
    s.set_state(V0, bxs, bys)

    t0 = time.time()
    save_snapshot(s, 0, outdir)
    dt_free, nstep = FIRST_DT, 0
    for k, tout in enumerate(times[1:], start=1):
        while s.t < tout - 1e-13:
            dt_cfl = s.cfl_dt()
            if nstep > 0:
                dt_free = min(dt_cfl, DT_GROWTH * dt_free)
            dt = min(dt_free, tout - s.t)
            s.step(dt)
            nstep += 1
            if nstep % 50 == 0:
                print(f"  paso {nstep:5d}  t={s.t:.5f}  dt={dt:.3e}  ({time.time()-t0:.0f} s)", flush=True)
        save_snapshot(s, k, outdir)
    print(f"Terminado: {nstep} pasos, {time.time()-t0:.1f} s. Salida en {outdir}")


if __name__ == "__main__":
    main()
