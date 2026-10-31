"""
tests_solver.py - Pruebas de verificacion del solver antes de correr el rotor.
  1) HLLD consistente: wl = wr  ->  flujo fisico
  2) Onda de Alfven circular (solucion exacta no lineal), orientada en x e y: convergencia ~ 2
  3) Simetria x <-> y del tubo de choque de Brio & Wu (prueba el barrido en y y la rotacion)
"""
import numpy as np
from mhd_core import *

GAM = 1.4


def test_hlld_consistency():
    S = np.empty((8, 8)); F = np.empty(8)
    w = np.array([1.3, 0.4, -0.2, 0.7, 0.8, 0.5, -0.6, 0.9])
    hlld(w, w, F, S, GAM)
    rho, vx, vy, vz, bx, by, bz, p = w
    B2 = bx*bx + by*by + bz*bz
    pt = p + 0.5*B2
    en = p/(GAM-1) + 0.5*rho*(vx*vx+vy*vy+vz*vz) + 0.5*B2
    Fe = np.empty(8); fill_F(Fe, rho, vx, vy, vz, bx, by, bz, pt, en)
    err = np.abs(F - Fe).max()
    print(f"[1] HLLD consistencia: max|F - F_fisico| = {err:.2e}")
    return err < 1e-12


def alfven(N, direction, delta=0.1):
    """Onda de Alfven circularmente polarizada, un periodo. direction 'x' o 'y'."""
    rho0, p0, b0 = 1.0, 0.1, 1.0
    c = b0/np.sqrt(rho0)
    L = 1.0
    n = 4
    if direction == 'x':
        s = MHDSolver2D(N, n, (0, L), (0, 1.0*n/N*L), gamma=GAM, cfl=0.4, bc="periodic")
        phase = lambda x: 2*np.pi*x/L
    else:
        s = MHDSolver2D(n, N, (0, 1.0*n/N*L), (0, L), gamma=GAM, cfl=0.4, bc="periodic")
        phase = lambda y: 2*np.pi*y/L

    def exact(t):
        X, Y = np.meshgrid(s.x, s.y, indexing="ij")
        ph = phase(X if direction == 'x' else Y) - 2*np.pi*c*t/L
        a, b = delta*np.cos(ph), delta*np.sin(ph)
        V = np.zeros((8, s.nx, s.ny)); V[0] = rho0; V[7] = p0
        if direction == 'x':       # normal x; (t1, t2) = (y, z)
            V[2], V[3] = a, b; V[5], V[6] = -np.sqrt(rho0)*a, -np.sqrt(rho0)*b; V[4] = b0
        else:                      # normal y; (t1, t2) = (z, x)
            V[3], V[1] = a, b; V[6], V[4] = -np.sqrt(rho0)*a, -np.sqrt(rho0)*b; V[5] = b0
        return V

    V0 = exact(0.0)
    if direction == 'x':
        bxs = np.full((s.nx+1, s.ny), b0)
        bys = np.zeros((s.nx, s.ny+1)); bys[:] = V0[5][:, :1]
    else:
        bys = np.full((s.nx, s.ny+1), b0)
        bxs = np.zeros((s.nx+1, s.ny)); bxs[:] = V0[4][:1, :]
    s.set_state(V0, bxs, bys)
    tend = L/c
    while s.t < tend - 1e-14:
        dt = min(s.cfl_dt(), tend - s.t)
        s.step(dt)
    V = s.primitives(); Ve = exact(tend)
    return np.mean([np.abs(V[k]-Ve[k]).mean() for k in (1, 2, 3, 4, 5, 6)]), s.max_divB()


def test_alfven():
    ok = True
    for d in ('x', 'y'):
        errs = []
        for N in (16, 32, 64, 128):
            e, dv = alfven(N, d)
            errs.append(e)
        order = np.log2(np.array(errs[:-1])/np.array(errs[1:]))
        print(f"[2] Alfven en {d}: L1 = " + ", ".join(f"{e:.3e}" for e in errs)
              + f"  | orden = " + ", ".join(f"{o:.2f}" for o in order) + f"  | max|divB| = {dv:.1e}")
        ok &= order[-1] > 1.7
    return ok


def briowu(direction, N=200, tend=0.1):
    """
    Brio & Wu (gamma=2) con Bz != 0. 'x': normal x. 'y': el mismo problema rotado 90 grados
    EN EL PLANO (x->y, y->-x, z->z), de modo que la componente evolucionada por CT sea la misma.
    """
    n = 4
    g = 2.0
    if direction == 'x':
        s = MHDSolver2D(N, n, (-1, 1), (0, 2.0*n/N), gamma=g, cfl=0.4, bc="outflow")
    else:
        s = MHDSolver2D(n, N, (0, 2.0*n/N), (-1, 1), gamma=g, cfl=0.4, bc="outflow")
    sh = (s.nx, s.ny)
    coord = s.x if direction == 'x' else s.y
    left = (coord < 0)
    sl = (slice(None), None) if direction == 'x' else (None, slice(None))
    B = lambda a: np.broadcast_to(np.where(left, a[0], a[1])[sl], sh)
    V = np.zeros((8, *sh)); bn = 0.75
    V[0] = B((1.0, 0.125)); V[7] = B((1.0, 0.1))
    bt, bz = B((1.0, -1.0)), B((0.5, -0.5))
    V[6] = bz
    if direction == 'x':
        V[4] = bn; V[5] = bt
        bxs = np.full((s.nx+1, s.ny), bn)
        bys = np.zeros((s.nx, s.ny+1)); bys[:] = bt[:, :1]
    else:
        V[5] = bn; V[4] = -bt
        bys = np.full((s.nx, s.ny+1), bn)
        bxs = np.zeros((s.nx+1, s.ny)); bxs[:] = -bt[:1, :]
    s.set_state(V, bxs, bys)
    while s.t < tend - 1e-14:
        s.step(min(s.cfl_dt(), tend - s.t))
    return s.primitives(), s


def test_briowu():
    Wx, sx = briowu('x'); Wy, sy = briowu('y')
    px = Wx[:, :, 1]
    py = Wy[:, 1, :]
    # x-caso [rho,vx,vy,vz,Bx,By,Bz,p]  <->  y-caso [rho, vy, -vx, vz, By, -Bx, Bz, p]
    mx = np.array([px[0], px[1], px[2], px[3], px[4], px[5], px[6], px[7]])
    my = np.array([py[0], py[2], -py[1], py[3], py[5], -py[4], py[6], py[7]])
    err = np.abs(mx - my).max()
    print(f"[3] Brio-Wu x vs y (rotado 90 grados): max|diferencia| = {err:.2e};  "
          f"rho in [{px[0].min():.4f}, {px[0].max():.4f}], max|vz|={abs(px[3]).max():.3f}, "
          f"max|divB| = {sx.max_divB():.1e}")
    return err < 1e-10


if __name__ == "__main__":
    r1 = test_hlld_consistency()
    r2 = test_alfven()
    r3 = test_briowu()
    print("RESULTADO:", "TODO OK" if (r1 and r2 and r3) else f"FALLA  {r1,r2,r3}")
