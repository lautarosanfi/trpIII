"""Versión discreta del juego, resuelta exactamente por programación lineal.

Por qué
-------
La deducción analítica arranca suponiendo estrategias «de umbral». Para no depender de
ese supuesto, acá se resuelve un juego finito que aproxima al original: los cuatro
números (X, Y, Z, W) se sortean uniformemente en la grilla de N puntos medios
{(k + 1/2)/N : k = 0..N-1} y los empates se reparten 1/2 y 1/2.

En ese juego finito se permiten **todas** las estrategias: J1 elige una distribución
sobre {P, C, D} para cada uno de sus N números (faroles, mezclas, usos raros de P y C,
etc.) y J2 elige una distribución sobre {P, C, D} para cada par (anuncio, número).
Como el juego es de suma constante, su equilibrio se obtiene resolviendo un programa
lineal (forma secuencial de Koller–Megiddo–von Stengel). Si la solución exacta del
juego finito muestra umbrales que tienden a s* y t* cuando N crece, el supuesto
inicial queda confirmado por un camino independiente.

Programa de J1 (maximiza):
    max  Σ_{h,j} p_j v[h,j]
    s.a. v[h,j] <= Σ_{a∈h} Σ_i p_i r[i,a] U_ab[i,j]     para toda acción b de J2
         Σ_a r[i,a] = 1,  r >= 0
donde h recorre los «conjuntos de información» de J2 (un anuncio, o uno solo si el
anuncio es secreto) y U_ab[i,j] = P(J1 gana | x_i, a, z_j, b).

Programa de J2 (minimiza):
    min  Σ_i p_i w[i]
    s.a. w[i] >= Σ_j Σ_b p_j q[h(a),j,b] U_ab[i,j]     para toda acción a de J1
         Σ_b q[h,j,b] = 1,  q >= 0
Por dualidad ambos valores coinciden (teorema minimax).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .modelo import COMPLEMENTO, DESCARTAR, PLANTARSE

VARIANTES = {
    "original": dict(acciones=(PLANTARSE, COMPLEMENTO, DESCARTAR), ve_anuncio=True),
    "sin_complemento": dict(acciones=(PLANTARSE, DESCARTAR), ve_anuncio=True),
    "anuncio_oculto": dict(acciones=(PLANTARSE, COMPLEMENTO, DESCARTAR), ve_anuncio=False),
}


def grilla(n: int) -> np.ndarray:
    return (np.arange(n) + 0.5) / n


def matriz_pago(a: int, b: int, n: int) -> np.ndarray:
    """U[i, j] = P(J1 gana | x = x_i, J1 hace a, z = z_j, J2 hace b) en el juego discreto.

    Los números finales se representan por su índice en la grilla (comparaciones exactas):
    plantarse -> i, complemento -> n-1-i. Si alguien descarta, su número es uniforme en la grilla.
    """
    k = np.arange(n)

    def indice(accion):
        if accion == PLANTARSE:
            return k
        if accion == COMPLEMENTO:
            return n - 1 - k
        return None

    i1, i2 = indice(a), indice(b)
    if i1 is not None and i2 is not None:
        A, B = i1[:, None], i2[None, :]
        return (A > B) + 0.5 * (A == B)
    if i1 is not None:   # J2 descarta: P(W < v) + P(W = v)/2 = (índice + 1/2)/n
        return np.repeat(((i1 + 0.5) / n)[:, None], n, axis=1)
    if i2 is not None:   # J1 descarta
        return np.repeat((1 - (i2 + 0.5) / n)[None, :], n, axis=0)
    return np.full((n, n), 0.5)


@dataclass
class SolucionDiscreta:
    n: int
    variante: str
    valor: float              # P(J1 gana) en equilibrio, según el programa de J1
    valor_j2: float           # ídem según el programa de J2 (debe coincidir)
    estrategia_j1: np.ndarray  # (n, 3): probabilidades de P, C, D para cada x_i
    estrategia_j2: np.ndarray  # (3, n, 3) indexada por la acción de J1, o (1, n, 3) si no ve el anuncio
    extra: dict = field(default_factory=dict)

    @property
    def x(self) -> np.ndarray:
        return grilla(self.n)

    def region_descarte_j1(self):
        """(mínimo, máximo) de los x en los que J1 descarta con probabilidad >= 1/2."""
        return _region(self.x, self.estrategia_j1[:, DESCARTAR])

    def region_descarte_j2(self, h: int):
        return _region(self.x, self.estrategia_j2[h, :, DESCARTAR])

    def celdas_mixtas_j1(self, tol: float = 1e-6) -> int:
        return int(((self.estrategia_j1 > tol).sum(axis=1) > 1).sum())


def _region(x, prob_descartar):
    sel = x[prob_descartar >= 0.5]
    return (float(sel.min()), float(sel.max())) if len(sel) else None


def _conjuntos_j2(acciones, ve_anuncio):
    """Conjuntos de información de J2: qué acciones de J1 caen en cada uno."""
    if ve_anuncio:
        return [[a] for a in acciones], {a: acciones.index(a) for a in acciones}
    return [list(acciones)], {a: 0 for a in acciones}


def resolver(n: int, variante: str = "original") -> SolucionDiscreta:
    """Equilibrio exacto del juego discreto con n puntos de grilla (ambos programas lineales)."""
    cfg = VARIANTES[variante]
    acc = cfg["acciones"]
    na = len(acc)
    p = np.full(n, 1.0 / n)
    U = {(a, b): matriz_pago(a, b, n) for a in acc for b in acc}
    conjuntos, h_de = _conjuntos_j2(acc, cfg["ve_anuncio"])
    H = len(conjuntos)
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

    # ----------------------------- programa de J1 -----------------------------
    nr, nv = n * na, H * n
    c = np.zeros(nr + nv)
    c[nr:] = -np.tile(p, H)
    filas, cols, vals, f = [], [], [], 0
    for h in range(H):
        for b in acc:
            filas.append(f + np.arange(n)); cols.append(nr + h * n + np.arange(n)); vals.append(np.ones(n))
            for a in conjuntos[h]:
                ia = acc.index(a)
                filas.append(f + J.ravel()); cols.append(I.ravel() * na + ia)
                vals.append(-(p[:, None] * U[a, b]).ravel())
            f += n
    A_ub = coo_matrix((np.concatenate(vals), (np.concatenate(filas), np.concatenate(cols))),
                      shape=(f, nr + nv)).tocsr()
    A_eq = coo_matrix((np.ones(nr), (np.repeat(np.arange(n), na), np.arange(nr))), shape=(n, nr + nv)).tocsr()
    res1 = linprog(c, A_ub=A_ub, b_ub=np.zeros(f), A_eq=A_eq, b_eq=np.ones(n),
                   bounds=[(0, 1)] * nr + [(None, None)] * nv, method="highs")
    if not res1.success:
        raise RuntimeError(f"programa de J1: {res1.message}")
    r = np.zeros((n, 3))
    r[:, list(acc)] = res1.x[:nr].reshape(n, na)

    # ----------------------------- programa de J2 -----------------------------
    nq = H * n * na
    c = np.zeros(nq + n)
    c[nq:] = p
    filas, cols, vals, f = [], [], [], 0
    for a in acc:
        h = h_de[a]
        for b in acc:
            ib = acc.index(b)
            filas.append(f + I.ravel()); cols.append((h * n + J.ravel()) * na + ib)
            vals.append((U[a, b] * p[None, :]).ravel())
        filas.append(f + np.arange(n)); cols.append(nq + np.arange(n)); vals.append(-np.ones(n))
        f += n
    A_ub = coo_matrix((np.concatenate(vals), (np.concatenate(filas), np.concatenate(cols))),
                      shape=(f, nq + n)).tocsr()
    A_eq = coo_matrix((np.ones(nq), (np.repeat(np.arange(H * n), na), np.arange(nq))),
                      shape=(H * n, nq + n)).tocsr()
    res2 = linprog(c, A_ub=A_ub, b_ub=np.zeros(f), A_eq=A_eq, b_eq=np.ones(H * n),
                   bounds=[(0, 1)] * nq + [(None, None)] * n, method="highs")
    if not res2.success:
        raise RuntimeError(f"programa de J2: {res2.message}")
    q = _por_accion(res2.x[:nq].reshape(H, n, na), acc, h_de, cfg["ve_anuncio"])

    return SolucionDiscreta(n, variante, float(-res1.fun), float(res2.fun), r, q)


def _por_accion(q_h, acc, h_de, ve_anuncio):
    """Reordena la estrategia de J2 para indexarla por la acción real de J1 (P=0, C=1, D=2).

    Si J2 ve el anuncio, la salida es (3, n, 3); los anuncios imposibles en la variante
    (p. ej. «complemento» si no existe) quedan en cero. Si no lo ve, es (1, n, 3).
    """
    H, n, na = q_h.shape
    if not ve_anuncio:
        q = np.zeros((1, n, 3))
        q[0][:, list(acc)] = q_h[0]
        return q
    q = np.zeros((3, n, 3))
    for a in acc:
        q[a][:, list(acc)] = q_h[h_de[a]]
    return q


# ---------------------------------------------------------------------------
# Juego ficticio (Brown, 1951; convergencia en suma cero: Robinson, 1951)
# ---------------------------------------------------------------------------
def juego_ficticio(n: int, iteraciones: int, variante: str = "original", registrar: int = 60):
    """Cada ronda, cada jugador responde de forma óptima al **promedio** de lo que el otro jugó antes.

    Se arranca de estrategias uniformes (todas las acciones con prob. 1/3). Devuelve un dict con
    la brecha de explotabilidad (valor de la mejor respuesta contra el promedio de J2 menos el
    de la mejor respuesta contra el promedio de J1; es >= 0 y vale 0 sólo en equilibrio), la
    estimación del valor y la estrategia promedio final de cada jugador.
    """
    cfg = VARIANTES[variante]
    acc = cfg["acciones"]
    na = len(acc)
    p = np.full(n, 1.0 / n)
    U = {(a, b): matriz_pago(a, b, n) for a in acc for b in acc}
    conjuntos, h_de = _conjuntos_j2(acc, cfg["ve_anuncio"])
    H = len(conjuntos)

    r_prom = np.full((n, na), 1.0 / na)
    q_prom = np.full((H, n, na), 1.0 / na)

    def pagos_j1(q):  # (n, na): pago de J1 en (x_i, a) contra q
        out = np.zeros((n, na))
        for ia, a in enumerate(acc):
            for ib, b in enumerate(acc):
                out[:, ia] += U[a, b] @ (p * q[h_de[a], :, ib])
        return out

    def pagos_j2(r):  # (H, n, na): pago de J1 en (h, z_j, b) contra r (J2 lo minimiza)
        out = np.zeros((H, n, na))
        for ia, a in enumerate(acc):
            for ib, b in enumerate(acc):
                out[h_de[a], :, ib] += (p * r[:, ia]) @ U[a, b]
        return out

    pasos = np.unique(np.geomspace(1, iteraciones, registrar).astype(int))
    hist = {"iteracion": [], "brecha": [], "valor_sup": [], "valor_inf": []}
    for k in range(1, iteraciones + 1):
        V1 = pagos_j1(q_prom)
        V2 = pagos_j2(r_prom)
        if k in pasos:
            sup = float(p @ V1.max(axis=1))
            inf = float((V2.min(axis=2) * p[None, :]).sum())
            hist["iteracion"].append(k); hist["brecha"].append(sup - inf)
            hist["valor_sup"].append(sup); hist["valor_inf"].append(inf)
        br1 = np.eye(na)[V1.argmax(axis=1)]
        br2 = np.eye(na)[V2.argmin(axis=2)]
        r_prom += (br1 - r_prom) / (k + 1)
        q_prom += (br2 - q_prom) / (k + 1)

    r = np.zeros((n, 3))
    r[:, list(acc)] = r_prom
    q = _por_accion(q_prom, acc, h_de, cfg["ve_anuncio"])
    return {k: np.array(v) for k, v in hist.items()} | {"estrategia_j1": r, "estrategia_j2": q}
