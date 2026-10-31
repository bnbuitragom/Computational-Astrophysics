#!/usr/bin/env python3
"""
convergence.py - Estudio de convergencia en malla del MHD Rotor (PLUTO y Python).

Como el rotor tiene choques y no existe solucion exacta, se usa autoconvergencia: la solucion
de referencia es la de PLUTO a la mayor resolucion disponible (p. ej. 800^2), promediada por
bloques (el promedio sobre las celdas finas contenidas en cada celda gruesa) hasta la malla N.
Para cada N < N_ref se mide, en t = 0.15:
   e_PLUTO(N)  = <|PLUTO_N  - ref_N|>      e_Python(N) = <|Python_N - ref_N|>
   d(N)        = <|Python_N - PLUTO_N|>    (diferencia entre codigos a igual resolucion)
donde <.> es el promedio sobre el dominio (area 1 -> integral). Se espera e ~ N^-p con p ~ 1
cerca de choques/discontinuidades. Si d(N) decae como e(N), los codigos solo difieren en error
de truncamiento; si d(N) se estanca, hay una diferencia sistematica entre los esquemas.

Directorios esperados (dentro de 08_Proyecto2):  pluto_run_{N}/  y  python_run_N{N}/
con fields_000k.npz (usar export_pluto.py para PLUTO y run_rotor.py -n N --times 0 0.15 para Python).
"""
import csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VARS = [("rho", r"$\rho$"), ("prs", r"$p$"), ("vx1", r"$v_x$"), ("vx2", r"$v_y$"),
        ("Bx1", r"$B_x$"), ("Bx2", r"$B_y$")]


def load_final(d, t_target=0.15, tol=1e-6):
    """Devuelve el snapshot con t = t_target del directorio d, o None si no existe."""
    d = Path(d)
    best = None
    for p in sorted(d.glob("fields_*.npz")):
        z = np.load(p)
        if abs(float(z["t"]) - t_target) < tol:
            best = {k: z[k] for k in z.files}
    return best


def restrict(a, f):
    """Promedio por bloques f x f (valido para volumenes finitos centrados en celda)."""
    n = a.shape[0]
    return a.reshape(n // f, f, n // f, f).mean(axis=(1, 3))


def collect(Ns=(100, 200, 400, 800), root=ROOT):
    """Carga los campos finales disponibles. Devuelve {('pluto'|'python', N): campos}."""
    data = {}
    for N in Ns:
        for code, pat in (("pluto", f"pluto_run_{N}"), ("python", f"python_run_N{N}")):
            f = load_final(Path(root) / pat)
            if f is not None and f["rho"].shape[0] == N:
                data[(code, N)] = f
    return data


def analyze(data):
    """Calcula errores. Referencia = PLUTO a la mayor N disponible. Devuelve (filas, N_ref)."""
    pl = sorted(N for (c, N) in data if c == "pluto")
    if len(pl) < 2:
        return [], (pl[-1] if pl else None)
    Nref = pl[-1]
    ref = data[("pluto", Nref)]
    rows = []
    for N in sorted({N for (_, N) in data if N < Nref and Nref % N == 0}):
        fac = Nref // N
        for key, _ in VARS:
            r = restrict(ref[key], fac)
            rec = dict(N=N, var=key)
            nr = np.abs(r).sum()
            if ("pluto", N) in data:
                a = data[("pluto", N)][key]
                rec["e_pluto"] = np.abs(a - r).mean()
                rec["e_pluto_rel"] = np.abs(a - r).sum() / nr
            if ("python", N) in data:
                b = data[("python", N)][key]
                rec["e_python"] = np.abs(b - r).mean()
                rec["e_python_rel"] = np.abs(b - r).sum() / nr
            if ("pluto", N) in data and ("python", N) in data:
                rec["d_py_pl"] = np.abs(b - a).mean()
                rec["d_py_pl_rel"] = np.abs(b - a).sum() / np.abs(a).sum()
            rows.append(rec)
    return rows, Nref


def observed_order(rows, var, col):
    pts = [(r["N"], r[col]) for r in rows if r["var"] == var and col in r and r[col] > 0]
    if len(pts) < 2:
        return np.nan
    N, e = np.array(pts).T
    return -np.polyfit(np.log(N), np.log(e), 1)[0]


def save_csv(rows, path):
    keys = ["N", "var", "e_pluto", "e_python", "d_py_pl", "e_pluto_rel", "e_python_rel", "d_py_pl_rel"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader(); w.writerows(rows)


def plot(rows, Nref, out=None):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (key, lab) in zip(axs.ravel(), VARS):
        rr = sorted([r for r in rows if r["var"] == key], key=lambda r: r["N"])
        for col, st, name in (("e_pluto", "ko-", "PLUTO vs ref"), ("e_python", "rs--", "Python vs ref"),
                              ("d_py_pl", "b^:", "|Python - PLUTO|")):
            pts = [(r["N"], r[col]) for r in rr if col in r and r[col] > 0]
            if pts:
                N, e = np.array(pts).T
                p = observed_order(rows, key, col)
                ax.loglog(N, e, st, label=f"{name} (p = {p:.2f})")
        Ns = np.array([r["N"] for r in rr], float)
        if Ns.size:
            e0 = next((r["e_pluto"] for r in rr if "e_pluto" in r), None)
            if e0:
                ax.loglog(Ns, e0 * (Ns / Ns[0])**-1, "gray", lw=0.8, ls="-.", label=r"$\propto N^{-1}$")
                ax.loglog(Ns, e0 * (Ns / Ns[0])**-2, "gray", lw=0.8, ls=":", label=r"$\propto N^{-2}$")
        ax.set_title(f"{lab}   (ref: PLUTO {Nref}$^2$)")
        ax.set_xlabel("N (celdas por lado)"); ax.set_ylabel(r"$\langle |\Delta| \rangle$")
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=130)
    return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    data = collect()
    print("Corridas disponibles:", sorted(data.keys()))
    rows, Nref = analyze(data)
    if not rows:
        raise SystemExit("Se necesitan al menos 2 resoluciones de PLUTO (la mayor es la referencia).")
    (ROOT / "reporte").mkdir(exist_ok=True); (ROOT / "figuras").mkdir(exist_ok=True)
    save_csv(rows, ROOT / "reporte" / "convergencia.csv")
    plot(rows, Nref, ROOT / "figuras" / "convergencia.png")
    for key, _ in VARS:
        print(f"{key:>4}: orden PLUTO = {observed_order(rows, key, 'e_pluto'):.2f}  "
              f"Python = {observed_order(rows, key, 'e_python'):.2f}  "
              f"dif. Py-PLUTO = {observed_order(rows, key, 'd_py_pl'):.2f}")
