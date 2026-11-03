"""Reglas del juego, estrategias y simulador Monte Carlo.

Reglas (enunciado de la cátedra)
--------------------------------
1. J1 recibe X ~ U(0,1).
2. J1 elige una de tres acciones:
     plantarse    -> se queda con X
     complemento  -> se queda con 1 - X
     descartar    -> se queda con un número nuevo Y ~ U(0,1)  (no vuelve a elegir)
3. J1 anuncia la acción elegida, pero no el número.
4-5. J2 recibe Z ~ U(0,1), conoce el anuncio de J1 y elige del mismo modo
     (se queda con Z, 1 - Z o un nuevo W ~ U(0,1)).
6. Gana el número final más alto (los empates tienen probabilidad 0).

Representación de estrategias
-----------------------------
Toda estrategia es *de comportamiento*: en cada situación devuelve la probabilidad
de cada acción, como un vector (P, C, D) que suma 1. Las estrategias puras son el
caso particular con probabilidades 0/1 y los «faroles» mixtos también entran.
La misma representación sirve para simular (se muestrea la acción) y para calcular
probabilidades exactas (se integra), así que simulación y cálculo usan exactamente
las mismas estrategias.

    J1: probs(x)          -> array (n, 3)
    J2: probs(anuncio, z) -> array (n, 3)   (anuncio = acción de J1, u OCULTO)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

PLANTARSE, COMPLEMENTO, DESCARTAR = 0, 1, 2
OCULTO = 3  # lo que «ve» J2 en la variante en que el anuncio de J1 es secreto
ACCIONES = (PLANTARSE, COMPLEMENTO, DESCARTAR)
NOMBRE_ACCION = {
    PLANTARSE: "plantarse",
    COMPLEMENTO: "complemento",
    DESCARTAR: "descartar",
    OCULTO: "(oculto)",
}
LETRA_ACCION = {PLANTARSE: "P", COMPLEMENTO: "C", DESCARTAR: "D"}


# ---------------------------------------------------------------------------
# Reglas
# ---------------------------------------------------------------------------
def numero_final(accion, original, nuevo):
    """Número con el que termina un jugador según la acción elegida.

    plantarse -> original;  complemento -> 1 - original;  descartar -> nuevo.
    """
    accion = np.asarray(accion)
    return np.where(
        accion == PLANTARSE,
        original,
        np.where(accion == COMPLEMENTO, 1.0 - original, nuevo),
    )


def muestrear(probs, rng):
    """Muestrea una acción por fila de una matriz de probabilidades (n, 3)."""
    probs = np.asarray(probs, dtype=float)
    acum = np.cumsum(probs, axis=-1)
    u = rng.random(probs.shape[:-1])
    # acción = cantidad de probabilidades acumuladas que u alcanza o supera
    return (u[..., None] >= acum[..., :-1]).sum(axis=-1)


def _mayor_de_dos(u):
    """Probabilidades de la acción que deja max(u, 1-u): plantarse si u >= 1/2, si no complemento."""
    u = np.asarray(u, dtype=float)
    p = np.zeros(u.shape + (3,))
    p[..., PLANTARSE] = u >= 0.5
    p[..., COMPLEMENTO] = u < 0.5
    return p


# ---------------------------------------------------------------------------
# Estrategias
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EstrategiaJ1:
    """Estrategia de comportamiento de J1: x -> probabilidades de (P, C, D)."""

    nombre: str
    probs: Callable[[np.ndarray], np.ndarray]

    def actuar(self, x, rng):
        return muestrear(self.probs(np.asarray(x, dtype=float)), rng)


@dataclass(frozen=True)
class EstrategiaJ2:
    """Estrategia de comportamiento de J2: (anuncio de J1, z) -> probabilidades de (P, C, D)."""

    nombre: str
    probs: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def actuar(self, anuncio, z, rng):
        return muestrear(self.probs(np.asarray(anuncio), np.asarray(z, dtype=float)), rng)


def j1_umbral(s: float) -> EstrategiaJ1:
    """Quedarse con max(x, 1-x) si es >= s; si no, descartar.

    s = 1/2 es la estrategia «ingenua» (complementar si x < 1/2, nunca descartar).
    """

    def probs(x):
        p = _mayor_de_dos(x)
        p[np.maximum(x, 1 - x) < s] = (0.0, 0.0, 1.0)
        return p

    return EstrategiaJ1(f"J1 umbral s={s:.4f}", probs)


def j1_farol_aleatorio(s: float, q: float) -> EstrategiaJ1:
    """Como j1_umbral(s), pero con max(x,1-x) < s se planta igual (con el mayor) con probabilidad q."""

    def probs(x):
        p = _mayor_de_dos(x)
        debil = np.maximum(x, 1 - x) < s
        p[debil] *= q
        p[debil, DESCARTAR] = 1 - q
        return p

    return EstrategiaJ1(f"J1 umbral {s:.4f} + farol con prob. {q:.2f}", probs)


def j1_farol_polarizado(s: float, b: float) -> EstrategiaJ1:
    """Se planta (con el mayor) si max(x,1-x) >= s **o** si max(x,1-x) < b; descarta en [b, s).

    Es el farol «con las peores manos»: anunciar fuerza justamente con los números más flojos.
    """

    def probs(x):
        p = _mayor_de_dos(x)
        m = np.maximum(x, 1 - x)
        p[(m >= b) & (m < s)] = (0.0, 0.0, 1.0)
        return p

    return EstrategiaJ1(f"J1 umbral {s:.4f} + farol si max<{b:.3f}", probs)


def j1_umbrales_por_anuncio(s_p: float, s_c: float) -> EstrategiaJ1:
    """Plantarse si x >= s_p; complemento si x < 1/2 y 1-x >= s_c; descartar en otro caso.

    Permite que «plantarse» y «complemento» transmitan información distinta (s_p != s_c).
    """

    def probs(x):
        p = np.zeros(np.shape(x) + (3,))
        p[..., PLANTARSE] = x >= max(s_p, 0.5)
        p[..., COMPLEMENTO] = (x < 0.5) & (1 - x >= s_c)
        p[..., DESCARTAR] = 1 - p[..., PLANTARSE] - p[..., COMPLEMENTO]
        return p

    return EstrategiaJ1(f"J1 umbrales P:{s_p:.3f} C:{s_c:.3f}", probs)


def j1_sin_complemento(s: float) -> EstrategiaJ1:
    """Variante sin complemento: plantarse si x >= s; si no, descartar."""

    def probs(x):
        p = np.zeros(np.shape(x) + (3,))
        p[..., PLANTARSE] = x >= s
        p[..., DESCARTAR] = x < s
        return p

    return EstrategiaJ1(f"J1 sin complemento, umbral {s:.4f}", probs)


def j1_tabla(probs_por_celda) -> EstrategiaJ1:
    """Estrategia constante por celdas: (0,1) dividido en N celdas iguales, probs_por_celda (N, 3)."""
    tabla = np.asarray(probs_por_celda, dtype=float)
    n = len(tabla)

    def probs(x):
        i = np.clip((np.asarray(x) * n).astype(int), 0, n - 1)
        return tabla[i].copy()

    return EstrategiaJ1(f"J1 tabla ({n} celdas)", probs)


def j2_umbrales(t_plantarse: float, t_descartar: float, t_complemento: float | None = None,
                nombre: str | None = None) -> EstrategiaJ2:
    """J2 se queda con max(z, 1-z) si alcanza el umbral asociado al anuncio de J1; si no, descarta."""
    t_c = t_plantarse if t_complemento is None else t_complemento
    tabla = np.array([t_plantarse, t_c, t_descartar], dtype=float)

    def probs(anuncio, z):
        anuncio = np.asarray(anuncio)
        if np.any(anuncio == OCULTO):
            raise ValueError("j2_umbrales necesita ver el anuncio de J1")
        p = _mayor_de_dos(z)
        p[np.maximum(z, 1 - z) < tabla[anuncio]] = (0.0, 0.0, 1.0)
        return p

    nombre = nombre or f"J2 umbrales P:{t_plantarse:.4f} C:{t_c:.4f} D:{t_descartar:.4f}"
    return EstrategiaJ2(nombre, probs)


def j2_sin_complemento(t_plantarse: float, t_descartar: float) -> EstrategiaJ2:
    """Variante sin complemento: J2 se planta si z alcanza el umbral del anuncio; si no, descarta."""
    tabla = np.array([t_plantarse, t_plantarse, t_descartar], dtype=float)

    def probs(anuncio, z):
        p = np.zeros(np.shape(z) + (3,))
        p[..., PLANTARSE] = z >= tabla[np.asarray(anuncio)]
        p[..., DESCARTAR] = 1 - p[..., PLANTARSE]
        return p

    return EstrategiaJ2(f"J2 sin complemento P:{t_plantarse:.4f} D:{t_descartar:.4f}", probs)


def j2_ignora_anuncio(t: float) -> EstrategiaJ2:
    """J2 usa el mismo umbral t sea cual sea el anuncio (también sirve si el anuncio es secreto)."""

    def probs(anuncio, z):
        p = _mayor_de_dos(z)
        p[np.maximum(z, 1 - z) < t] = (0.0, 0.0, 1.0)
        return p

    return EstrategiaJ2(f"J2 umbral único {t:.4f}", probs)


def j2_tabla(probs_por_anuncio_y_celda) -> EstrategiaJ2:
    """Estrategia de J2 constante por celdas. Forma (3, N, 3) si ve el anuncio, (1, N, 3) si no."""
    tabla = np.asarray(probs_por_anuncio_y_celda, dtype=float)
    n = tabla.shape[1]

    def probs(anuncio, z):
        h = np.zeros(np.shape(z), dtype=int) if tabla.shape[0] == 1 else np.asarray(anuncio)
        j = np.clip((np.asarray(z) * n).astype(int), 0, n - 1)
        return tabla[h, j].copy()

    return EstrategiaJ2(f"J2 tabla ({n} celdas)", probs)


# ---------------------------------------------------------------------------
# Simulación
# ---------------------------------------------------------------------------
@dataclass
class Partidas:
    """Resultado de n partidas simuladas (un elemento de cada array por partida)."""

    x: np.ndarray   # número original de J1
    a1: np.ndarray  # acción (= anuncio) de J1
    v1: np.ndarray  # número final de J1
    z: np.ndarray   # número original de J2
    a2: np.ndarray  # acción de J2
    v2: np.ndarray  # número final de J2

    @property
    def gana_j1(self) -> np.ndarray:
        return self.v1 > self.v2


def simular(e1: EstrategiaJ1, e2: EstrategiaJ2, n: int, rng: np.random.Generator,
            ve_anuncio: bool = True) -> Partidas:
    """Simula n partidas independientes siguiendo las reglas al pie de la letra."""
    x, y, z, w = rng.random((4, n))
    a1 = e1.actuar(x, rng)
    v1 = numero_final(a1, x, y)
    anuncio = a1 if ve_anuncio else np.full(n, OCULTO)
    a2 = e2.actuar(anuncio, z, rng)
    v2 = numero_final(a2, z, w)
    return Partidas(x, a1, v1, z, a2, v2)


def simular_j1(e1: EstrategiaJ1, n: int, rng: np.random.Generator):
    """Sólo el turno de J1: devuelve (acción anunciada, número final). Es lo que J2 puede observar
    (el anuncio durante la partida y el número al final) si juega muchas veces."""
    x, y = rng.random((2, n))
    a1 = e1.actuar(x, rng)
    return a1, numero_final(a1, x, y)


@dataclass(frozen=True)
class Estimacion:
    """Proporción estimada por Monte Carlo, con su error estándar."""

    p: float
    n: int

    @property
    def error_estandar(self) -> float:
        return float(np.sqrt(self.p * (1 - self.p) / self.n))

    @property
    def ic95(self) -> tuple[float, float]:
        d = 1.96 * self.error_estandar
        return (self.p - d, self.p + d)

    def __str__(self) -> str:
        return f"{self.p:.4f} ± {1.96 * self.error_estandar:.4f}"


def estimar_prob_j1(e1: EstrategiaJ1, e2: EstrategiaJ2, n: int, rng: np.random.Generator,
                    ve_anuncio: bool = True, lote: int = 1_000_000) -> Estimacion:
    """P(J1 gana) por Monte Carlo, simulando en lotes para no agotar memoria."""
    ganadas, hechas = 0, 0
    while hechas < n:
        m = min(lote, n - hechas)
        ganadas += int(simular(e1, e2, m, rng, ve_anuncio).gana_j1.sum())
        hechas += m
    return Estimacion(ganadas / n, n)
