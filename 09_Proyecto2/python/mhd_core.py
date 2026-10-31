"""
mhd_core.py - Solver MHD ideal 2.5D (Cartesiano) para el proyecto 2.

Replica la numerica de PLUTO (configuracion #01 del MHD Rotor):
  * 8 variables: rho, vx, vy, vz, Bx, By, Bz, p   (gamma-law, B^2/2 como presion magnetica)
  * Reconstruccion lineal de variables primitivas con limitador MC
  * Solver de Riemann HLLD (Miyoshi & Kusano 2005)
  * Constrained Transport (campo B en las caras) con EMF aritmetico en las esquinas
  * Integracion temporal RK2 (Heun):  U1 = U0 + dt L(U0);  Un = 0.5 U0 + 0.5 (U1 + dt L(U1))

Convenciones de indices (arrays [x, y], igual que pyPLUTO):
  U[v, i, j]   v = 0..7 : D, Mx, My, Mz, Bx(cc), By(cc), Bz(cc), E     (conservadas)
  V[v, i, j]   v = 0..7 : rho, vx, vy, vz, Bx, By, Bz, p               (primitivas)
  bxs[i, j]   Bx en la cara IZQUIERDA de la celda i   -> forma (nx+2g+1, ny+2g)
  bys[i, j]   By en la cara INFERIOR  de la celda j   -> forma (nx+2g, ny+2g+1)
  g = NG = 2 celdas fantasma. Celdas fisicas: i in [g, g+nx), j in [g, g+ny)
"""
import numpy as np
from numba import njit, prange

NG = 2
RHO, VX, VY, VZ, BX, BY, BZ, PR = 0, 1, 2, 3, 4, 5, 6, 7   # primitivas
ENE = 7                                                     # energia en U
# dt = CFL / max_celdas( max_dir(lambda_d/dx_d) )  (como se observa en PLUTO: ~589 pasos en 400x400).
# Con True se usa la suma sobre direcciones, mas conservadora (~2x mas pasos).
SUM_DIRECTIONS = False
P_FLOOR = 1.0e-10
R_FLOOR = 1.0e-10


# ----------------------------------------------------------------------------
# Utilidades de bajo nivel
# ----------------------------------------------------------------------------
@njit(inline="always")
def mc_limiter(a, b):
    """Limitador MC (monotonized central): min(2|a|, 2|b|, |a+b|/2) con signo."""
    if a * b <= 0.0:
        return 0.0
    c = 0.5 * (a + b)
    m = min(2.0 * abs(a), 2.0 * abs(b), abs(c))
    return m if c > 0.0 else -m


@njit(inline="always")
def fill_U(U, rho, vx, vy, vz, bx, by, bz, en):
    U[0] = rho
    U[1] = rho * vx
    U[2] = rho * vy
    U[3] = rho * vz
    U[4] = bx
    U[5] = by
    U[6] = bz
    U[7] = en


@njit(inline="always")
def fill_F(F, rho, vx, vy, vz, bx, by, bz, pt, en):
    """Flujo fisico en la direccion x (pt = presion total p + B^2/2)."""
    vb = vx * bx + vy * by + vz * bz
    F[0] = rho * vx
    F[1] = rho * vx * vx + pt - bx * bx
    F[2] = rho * vx * vy - bx * by
    F[3] = rho * vx * vz - bx * bz
    F[4] = 0.0
    F[5] = by * vx - bx * vy
    F[6] = bz * vx - bx * vz
    F[7] = (en + pt) * vx - bx * vb


@njit(cache=True)
def hlld(wl, wr, F, S, gam):
    """
    Flujo HLLD (Miyoshi & Kusano 2005) en la direccion x.
    wl, wr : estados primitivos [rho, vx, vy, vz, Bx, By, Bz, p]  (Bx identico en ambos)
    F      : salida, flujo de las conservadas [D, Mx, My, Mz, Bx, By, Bz, E]
    S      : buffer de trabajo (8, 8)
    """
    rL = wl[0]; uL = wl[1]; vL = wl[2]; wL = wl[3]; byL = wl[5]; bzL = wl[6]; pL = wl[7]
    rR = wr[0]; uR = wr[1]; vR = wr[2]; wR = wr[3]; byR = wr[5]; bzR = wr[6]; pR = wr[7]
    bx = wl[4]
    bx2 = bx * bx
    B2L = bx2 + byL * byL + bzL * bzL
    B2R = bx2 + byR * byR + bzR * bzR
    ptL = pL + 0.5 * B2L
    ptR = pR + 0.5 * B2R
    eL = pL / (gam - 1.0) + 0.5 * rL * (uL * uL + vL * vL + wL * wL) + 0.5 * B2L
    eR = pR / (gam - 1.0) + 0.5 * rR * (uR * uR + vR * vR + wR * wR) + 0.5 * B2R

    # velocidades magnetosonicas rapidas
    a2 = gam * pL / rL
    ct = a2 + B2L / rL
    cfL = np.sqrt(0.5 * (ct + np.sqrt(max(ct * ct - 4.0 * a2 * bx2 / rL, 0.0))))
    a2 = gam * pR / rR
    ct = a2 + B2R / rR
    cfR = np.sqrt(0.5 * (ct + np.sqrt(max(ct * ct - 4.0 * a2 * bx2 / rR, 0.0))))
    cmax = max(cfL, cfR)
    SL = min(uL, uR) - cmax
    SR = max(uL, uR) + cmax

    UL = S[0]; UR = S[1]; FL = S[2]; FR = S[3]
    UsL = S[4]; UsR = S[5]; Uss = S[6]
    fill_U(UL, rL, uL, vL, wL, bx, byL, bzL, eL)
    fill_U(UR, rR, uR, vR, wR, bx, byR, bzR, eR)
    fill_F(FL, rL, uL, vL, wL, bx, byL, bzL, ptL, eL)
    fill_F(FR, rR, uR, vR, wR, bx, byR, bzR, ptR, eR)

    if SL >= 0.0:
        for k in range(8):
            F[k] = FL[k]
        return
    if SR <= 0.0:
        for k in range(8):
            F[k] = FR[k]
        return

    dL = SL - uL
    dR = SR - uR
    SM = ((dR * rR * uR - dL * rL * uL) - ptR + ptL) / (dR * rR - dL * rL)
    pts = ptL + rL * dL * (SM - uL)
    tol = 1.0e-12 * (ptL + ptR)

    # ---- estado estrella izquierdo
    rsL = rL * dL / (SL - SM)
    denL = rL * dL * (SL - SM) - bx2
    if abs(denL) < tol:
        vsL = vL; wsL = wL; bysL = byL; bzsL = bzL
    else:
        vsL = vL - bx * byL * (SM - uL) / denL
        wsL = wL - bx * bzL * (SM - uL) / denL
        fac = (rL * dL * dL - bx2) / denL
        bysL = byL * fac
        bzsL = bzL * fac
    vbL = uL * bx + vL * byL + wL * bzL
    vbsL = SM * bx + vsL * bysL + wsL * bzsL
    esL = (dL * eL - ptL * uL + pts * SM + bx * (vbL - vbsL)) / (SL - SM)
    fill_U(UsL, rsL, SM, vsL, wsL, bx, bysL, bzsL, esL)

    # ---- estado estrella derecho
    rsR = rR * dR / (SR - SM)
    denR = rR * dR * (SR - SM) - bx2
    if abs(denR) < tol:
        vsR = vR; wsR = wR; bysR = byR; bzsR = bzR
    else:
        vsR = vR - bx * byR * (SM - uR) / denR
        wsR = wR - bx * bzR * (SM - uR) / denR
        fac = (rR * dR * dR - bx2) / denR
        bysR = byR * fac
        bzsR = bzR * fac
    vbR = uR * bx + vR * byR + wR * bzR
    vbsR = SM * bx + vsR * bysR + wsR * bzsR
    esR = (dR * eR - ptR * uR + pts * SM + bx * (vbR - vbsR)) / (SR - SM)
    fill_U(UsR, rsR, SM, vsR, wsR, bx, bysR, bzsR, esR)

    sqL = np.sqrt(rsL)
    sqR = np.sqrt(rsR)
    SsL = SM - abs(bx) / sqL
    SsR = SM + abs(bx) / sqR

    if SsL >= 0.0:                         # region *L
        for k in range(8):
            F[k] = FL[k] + SL * (UsL[k] - UL[k])
        return
    if SsR <= 0.0:                         # region *R
        for k in range(8):
            F[k] = FR[k] + SR * (UsR[k] - UR[k])
        return

    # ---- estado doble estrella (regiones **L y **R)
    sg = 1.0 if bx > 0.0 else -1.0
    isum = 1.0 / (sqL + sqR)
    vss = (sqL * vsL + sqR * vsR + (bysR - bysL) * sg) * isum
    wss = (sqL * wsL + sqR * wsR + (bzsR - bzsL) * sg) * isum
    byss = (sqL * bysR + sqR * bysL + sqL * sqR * (vsR - vsL) * sg) * isum
    bzss = (sqL * bzsR + sqR * bzsL + sqL * sqR * (wsR - wsL) * sg) * isum
    vbss = SM * bx + vss * byss + wss * bzss
    if SM >= 0.0:                          # region **L
        essL = esL - sqL * (vbsL - vbss) * sg
        fill_U(Uss, rsL, SM, vss, wss, bx, byss, bzss, essL)
        for k in range(8):
            F[k] = FL[k] + SL * (UsL[k] - UL[k]) + SsL * (Uss[k] - UsL[k])
    else:                                  # region **R
        essR = esR + sqR * (vbsR - vbss) * sg
        fill_U(Uss, rsR, SM, vss, wss, bx, byss, bzss, essR)
        for k in range(8):
            F[k] = FR[k] + SR * (UsR[k] - UR[k]) + SsR * (Uss[k] - UsR[k])


# ----------------------------------------------------------------------------
# Conversiones y paso de tiempo
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def cons_to_prim(U, V, gam):
    nxt, nyt = U.shape[1], U.shape[2]
    for i in prange(nxt):
        for j in range(nyt):
            d = max(U[0, i, j], R_FLOOR)
            vx = U[1, i, j] / d
            vy = U[2, i, j] / d
            vz = U[3, i, j] / d
            bx = U[4, i, j]; by = U[5, i, j]; bz = U[6, i, j]
            ek = 0.5 * d * (vx * vx + vy * vy + vz * vz)
            eb = 0.5 * (bx * bx + by * by + bz * bz)
            p = (gam - 1.0) * (U[7, i, j] - ek - eb)
            V[0, i, j] = d
            V[1, i, j] = vx; V[2, i, j] = vy; V[3, i, j] = vz
            V[4, i, j] = bx; V[5, i, j] = by; V[6, i, j] = bz
            V[7, i, j] = max(p, P_FLOOR)


def prim_to_cons(V, gam):
    U = np.empty_like(V)
    d = V[0]
    U[0] = d
    U[1] = d * V[1]; U[2] = d * V[2]; U[3] = d * V[3]
    U[4:7] = V[4:7]
    U[7] = (V[7] / (gam - 1.0) + 0.5 * d * (V[1]**2 + V[2]**2 + V[3]**2)
            + 0.5 * (V[4]**2 + V[5]**2 + V[6]**2))
    return U


@njit(parallel=True, cache=True)
def max_inv_dt(V, gam, dx, dy, ng, nx, ny):
    """max sobre celdas de  max_d (|v_d|+cf_d)/dx_d  (o su suma si SUM_DIRECTIONS)."""
    rowmax = np.zeros(nx)
    for ii in prange(nx):
        i = ii + ng
        loc = 0.0
        for j in range(ng, ng + ny):
            r = V[0, i, j]
            a2 = gam * V[7, i, j] / r
            B2 = V[4, i, j]**2 + V[5, i, j]**2 + V[6, i, j]**2
            ct = a2 + B2 / r
            cx = np.sqrt(0.5 * (ct + np.sqrt(max(ct * ct - 4.0 * a2 * V[4, i, j]**2 / r, 0.0))))
            cy = np.sqrt(0.5 * (ct + np.sqrt(max(ct * ct - 4.0 * a2 * V[5, i, j]**2 / r, 0.0))))
            if SUM_DIRECTIONS:
                val = (abs(V[1, i, j]) + cx) / dx + (abs(V[2, i, j]) + cy) / dy
            else:
                val = max((abs(V[1, i, j]) + cx) / dx, (abs(V[2, i, j]) + cy) / dy)
            if val > loc:
                loc = val
        rowmax[ii] = loc
    return rowmax.max()


# ----------------------------------------------------------------------------
# Barridos de flujo (reconstruccion MC + HLLD)
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def sweep_x(V, bxs, fx, gam, ng, nx, ny):
    """Flujos en las caras x: fx[:, i, j] = flujo en la cara izquierda de la celda i."""
    nxt = V.shape[1]
    for j in prange(ng - 1, ng + ny + 1):
        dq = np.zeros((8, nxt))
        wl = np.empty(8); wr = np.empty(8); F = np.empty(8); S = np.empty((8, 8))
        for i in range(1, nxt - 1):
            for v in range(8):
                dq[v, i] = mc_limiter(V[v, i, j] - V[v, i - 1, j], V[v, i + 1, j] - V[v, i, j])
        for i in range(ng, ng + nx + 1):
            for v in range(8):
                wl[v] = V[v, i - 1, j] + 0.5 * dq[v, i - 1]
                wr[v] = V[v, i, j] - 0.5 * dq[v, i]
            if wl[0] <= 0.0 or wl[7] <= 0.0:          # respaldo de 1er orden
                for v in range(8):
                    wl[v] = V[v, i - 1, j]
            if wr[0] <= 0.0 or wr[7] <= 0.0:
                for v in range(8):
                    wr[v] = V[v, i, j]
            wl[4] = bxs[i, j]
            wr[4] = bxs[i, j]
            hlld(wl, wr, F, S, gam)
            for v in range(8):
                fx[v, i, j] = F[v]


@njit(parallel=True, cache=True)
def sweep_y(V, bys, fy, gam, ng, nx, ny):
    """
    Flujos en las caras y. Se rota (x,y,z)->(y,z,x) para reutilizar HLLD:
    normal=y, tangenciales=(z, x). fy[:, i, j] = flujo en la cara inferior de la celda j.
    """
    nyt = V.shape[2]
    for i in prange(ng - 1, ng + nx + 1):
        dq = np.zeros((8, nyt))
        pl = np.empty(8); pr = np.empty(8)
        wl = np.empty(8); wr = np.empty(8); F = np.empty(8); S = np.empty((8, 8))
        for j in range(1, nyt - 1):
            for v in range(8):
                dq[v, j] = mc_limiter(V[v, i, j] - V[v, i, j - 1], V[v, i, j + 1] - V[v, i, j])
        for j in range(ng, ng + ny + 1):
            for v in range(8):
                pl[v] = V[v, i, j - 1] + 0.5 * dq[v, j - 1]
                pr[v] = V[v, i, j] - 0.5 * dq[v, j]
            if pl[0] <= 0.0 or pl[7] <= 0.0:
                for v in range(8):
                    pl[v] = V[v, i, j - 1]
            if pr[0] <= 0.0 or pr[7] <= 0.0:
                for v in range(8):
                    pr[v] = V[v, i, j]
            # rotacion: [rho, vn=vy, vt1=vz, vt2=vx, Bn=By, Bt1=Bz, Bt2=Bx, p]
            wl[0] = pl[0]; wl[1] = pl[2]; wl[2] = pl[3]; wl[3] = pl[1]
            wl[4] = bys[i, j]; wl[5] = pl[6]; wl[6] = pl[4]; wl[7] = pl[7]
            wr[0] = pr[0]; wr[1] = pr[2]; wr[2] = pr[3]; wr[3] = pr[1]
            wr[4] = bys[i, j]; wr[5] = pr[6]; wr[6] = pr[4]; wr[7] = pr[7]
            hlld(wl, wr, F, S, gam)
            # de vuelta al orden fisico [D, Mx, My, Mz, Bx, By, Bz, E]
            fy[0, i, j] = F[0]
            fy[1, i, j] = F[3]
            fy[2, i, j] = F[1]
            fy[3, i, j] = F[2]
            fy[4, i, j] = F[6]     # = vy Bx - vx By = +Ez en la cara y
            fy[5, i, j] = F[4]     # = 0
            fy[6, i, j] = F[5]
            fy[7, i, j] = F[7]


# ----------------------------------------------------------------------------
# Actualizacion: flujos para las celdas + CT para el campo en las caras
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def update(Un, bxn, byn, fx, fy, dt, dx, dy, ng, nx, ny):
    cx = dt / dx
    cy = dt / dy
    # EMF en esquinas (lado inferior-izquierdo de la celda (ic,jc)); promedio aritmetico
    # Ez(cara x) = -F_x(By) = -fx[5];   Ez(cara y) = +F_y(Bx) = fy[4]
    ez = np.zeros((nx + 2 * ng + 1, ny + 2 * ng + 1))
    for ic in prange(ng, ng + nx + 1):
        for jc in range(ng, ng + ny + 1):
            ez[ic, jc] = 0.25 * (-fx[5, ic, jc - 1] - fx[5, ic, jc]
                                 + fy[4, ic - 1, jc] + fy[4, ic, jc])
    for i in prange(ng, ng + nx + 1):
        for j in range(ng, ng + ny):
            bxn[i, j] -= cy * (ez[i, j + 1] - ez[i, j])
    for i in prange(ng, ng + nx):
        for j in range(ng, ng + ny + 1):
            byn[i, j] += cx * (ez[i + 1, j] - ez[i, j])
    # variables de celda (D, M, Bz, E)
    for i in prange(ng, ng + nx):
        for j in range(ng, ng + ny):
            for v in (0, 1, 2, 3, 6, 7):
                Un[v, i, j] -= cx * (fx[v, i + 1, j] - fx[v, i, j]) \
                             + cy * (fy[v, i, j + 1] - fy[v, i, j])
            Un[4, i, j] = 0.5 * (bxn[i, j] + bxn[i + 1, j])
            Un[5, i, j] = 0.5 * (byn[i, j] + byn[i, j + 1])


# ----------------------------------------------------------------------------
# Condiciones de frontera
# ----------------------------------------------------------------------------
def apply_bc(U, bxs, bys, nx, ny, bc="outflow"):
    """Rellena celdas fantasma (en sitio). bc = 'outflow' (gradiente cero) o 'periodic'."""
    g = NG
    if bc == "outflow":
        U[:, :g, :] = U[:, g:g + 1, :]
        U[:, g + nx:, :] = U[:, g + nx - 1:g + nx, :]
        U[:, :, :g] = U[:, :, g:g + 1]
        U[:, :, g + ny:] = U[:, :, g + ny - 1:g + ny]
        bxs[:g, :] = bxs[g:g + 1, :]
        bxs[g + nx + 1:, :] = bxs[g + nx:g + nx + 1, :]
        bxs[:, :g] = bxs[:, g:g + 1]
        bxs[:, g + ny:] = bxs[:, g + ny - 1:g + ny]
        bys[:g, :] = bys[g:g + 1, :]
        bys[g + nx:, :] = bys[g + nx - 1:g + nx, :]
        bys[:, :g] = bys[:, g:g + 1]
        bys[:, g + ny + 1:] = bys[:, g + ny:g + ny + 1]
    elif bc == "periodic":
        U[:, :g, :] = U[:, nx:nx + g, :]
        U[:, g + nx:, :] = U[:, g:2 * g, :]
        U[:, :, :g] = U[:, :, ny:ny + g]
        U[:, :, g + ny:] = U[:, :, g:2 * g]
        bxs[g + nx, :] = bxs[g, :]
        bxs[:g, :] = bxs[nx:nx + g, :]
        bxs[g + nx + 1:, :] = bxs[g + 1:2 * g + 1, :]
        bxs[:, :g] = bxs[:, ny:ny + g]
        bxs[:, g + ny:] = bxs[:, g:2 * g]
        bys[:, g + ny] = bys[:, g]
        bys[:, :g] = bys[:, ny:ny + g]
        bys[:, g + ny + 1:] = bys[:, g + 1:2 * g + 1]
        bys[:g, :] = bys[nx:nx + g, :]
        bys[g + nx:, :] = bys[g:2 * g, :]
    else:
        raise ValueError(bc)
    # B centrado en celda = promedio de las caras (tambien en fantasmas)
    U[4] = 0.5 * (bxs[:-1, :] + bxs[1:, :])
    U[5] = 0.5 * (bys[:, :-1] + bys[:, 1:])


# ----------------------------------------------------------------------------
# Clase del solver
# ----------------------------------------------------------------------------
class MHDSolver2D:
    def __init__(self, nx, ny, xlim, ylim, gamma=1.4, cfl=0.4, bc="outflow"):
        self.nx, self.ny, self.gam, self.cfl, self.bc = nx, ny, gamma, cfl, bc
        self.dx = (xlim[1] - xlim[0]) / nx
        self.dy = (ylim[1] - ylim[0]) / ny
        self.x = xlim[0] + (np.arange(nx) + 0.5) * self.dx
        self.y = ylim[0] + (np.arange(ny) + 0.5) * self.dy
        g = NG
        self.U = np.zeros((8, nx + 2 * g, ny + 2 * g))
        self.bxs = np.zeros((nx + 2 * g + 1, ny + 2 * g))
        self.bys = np.zeros((nx + 2 * g, ny + 2 * g + 1))
        self.t = 0.0

    def set_state(self, V_int, bxs_int, bys_int):
        """V_int: (8,nx,ny) primitivas en centros; bxs_int: (nx+1,ny); bys_int: (nx,ny+1)."""
        g, nx, ny = NG, self.nx, self.ny
        V = np.zeros_like(self.U)
        V[:, g:g + nx, g:g + ny] = V_int
        V[0, :g, :] = V[0, :, :g] = V[0, g + nx:, :] = V[0, :, g + ny:] = 1.0
        V[7, :g, :] = V[7, :, :g] = V[7, g + nx:, :] = V[7, :, g + ny:] = 1.0
        self.bxs[g:g + nx + 1, g:g + ny] = bxs_int
        self.bys[g:g + nx, g:g + ny + 1] = bys_int
        V[4, g:g + nx, g:g + ny] = 0.5 * (bxs_int[:-1] + bxs_int[1:])
        V[5, g:g + nx, g:g + ny] = 0.5 * (bys_int[:, :-1] + bys_int[:, 1:])
        self.U[:] = prim_to_cons(V, self.gam)
        apply_bc(self.U, self.bxs, self.bys, nx, ny, self.bc)

    # ---- diagnosticos
    def primitives(self):
        V = np.empty_like(self.U)
        cons_to_prim(self.U, V, self.gam)
        g = NG
        return V[:, g:g + self.nx, g:g + self.ny].copy()

    def faces(self):
        g = NG
        return (self.bxs[g:g + self.nx + 1, g:g + self.ny].copy(),
                self.bys[g:g + self.nx, g:g + self.ny + 1].copy())

    def max_divB(self):
        bx, by = self.faces()
        d = (bx[1:] - bx[:-1]) / self.dx + (by[:, 1:] - by[:, :-1]) / self.dy
        return np.abs(d).max()

    def mass(self):
        g = NG
        return self.U[0, g:g + self.nx, g:g + self.ny].sum() * self.dx * self.dy

    def cfl_dt(self):
        V = np.empty_like(self.U)
        apply_bc(self.U, self.bxs, self.bys, self.nx, self.ny, self.bc)
        cons_to_prim(self.U, V, self.gam)
        return self.cfl / max_inv_dt(V, self.gam, self.dx, self.dy, NG, self.nx, self.ny)

    # ---- un paso de Euler hacia adelante: devuelve U + dt L(U)
    def _euler(self, U, bxs, bys, dt):
        nx, ny, g = self.nx, self.ny, NG
        apply_bc(U, bxs, bys, nx, ny, self.bc)
        V = np.empty_like(U)
        cons_to_prim(U, V, self.gam)
        fx = np.zeros_like(U)
        fy = np.zeros_like(U)
        sweep_x(V, bxs, fx, self.gam, g, nx, ny)
        sweep_y(V, bys, fy, self.gam, g, nx, ny)
        Un, bxn, byn = U.copy(), bxs.copy(), bys.copy()
        update(Un, bxn, byn, fx, fy, dt, self.dx, self.dy, g, nx, ny)
        return Un, bxn, byn

    def step(self, dt):
        """Un paso RK2 (Heun)."""
        U1, bx1, by1 = self._euler(self.U, self.bxs, self.bys, dt)
        U2, bx2, by2 = self._euler(U1, bx1, by1, dt)
        self.U = 0.5 * (self.U + U2)
        self.bxs = 0.5 * (self.bxs + bx2)
        self.bys = 0.5 * (self.bys + by2)
        self.t += dt
