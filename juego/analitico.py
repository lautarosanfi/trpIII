"""Fórmulas cerradas del juego.

Notación (la misma del entregable)
----------------------------------
M = max(u, 1-u) es el mejor número que un jugador puede quedarse sin descartar;
si u ~ U(0,1) entonces M ~ U(1/2, 1).

J1 «de umbral s»: se queda con M1 si M1 >= s; si no, descarta.
J2 «de umbrales (t, tau)»: tras un anuncio de plantarse/complemento se queda con M2
si M2 >= t; tras un descarte se queda con M2 si M2 >= tau; en otro caso descarta.

Resultados principales (demostrados en el entregable)
-----------------------------------------------------
  s* = 4^(-1/3)          (s*^3 = 1/4)
  t* = (1 + s*^2) / 2
  tau* = 1/2             (tras un descarte de J1, J2 nunca descarta)
  P(J1 gana) = 1/4 + 3 s*/8 ≈ 0.4862,   P(J2 gana) ≈ 0.5138
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Equilibrio del juego original
# ---------------------------------------------------------------------------
S_ESTRELLA = 4 ** (-1 / 3)
T_ESTRELLA = (1 + S_ESTRELLA**2) / 2
TAU_ESTRELLA = 0.5
P_GANA_J1 = 0.25 + 3 * S_ESTRELLA / 8
P_GANA_J2 = 1 - P_GANA_J1


def umbral_j2(s):
    """Mejor respuesta de J2 tras «plantarse/complemento» si J1 usa umbral s: t(s) = (1+s²)/2.

    Sale de igualar P(V1 < t | plantó) = (t-s)/(1-s) con P(gana descartando) = 1 - E[V1|plantó] = (1-s)/2.
    """
    return (1 + np.asarray(s, dtype=float) ** 2) / 2


def pago_plantarse(v, t):
    """P(J1 gana) si J1 termina con el número v (sin descartar) y J2, tras ese anuncio, usa umbral t.

    J2 descarta con prob. P(M2 < t) = 2t-1 (y entonces queda W ~ U(0,1)) o se queda con M2 ~ U(t,1):
        P(V2 < v) = (2t-1)·v + 2·(v-t)⁺.
    """
    v = np.asarray(v, dtype=float)
    return (2 * t - 1) * v + 2 * np.maximum(v - t, 0.0)


def pago_descartar(tau):
    """P(J1 gana) si J1 descarta y J2, tras el descarte, usa umbral tau:  ∫₀¹ P(V2 < y) dy = 1/2 - tau + tau²."""
    return 0.5 - tau + tau**2


def prob_j1(s, t, tau=0.5):
    """P(J1 gana) con J1 de umbral s y J2 de umbrales (t, tau). Válida para s, t, tau en [1/2, 1]."""
    return ((2 * s - 1) * pago_descartar(tau)
            + (2 * t - 1) * (1 - s**2)
            + 2 * (1 - t) ** 2
            - 2 * np.maximum(s - t, 0.0) ** 2)


def prob_j1_umbral(s):
    """g(s): P(J1 gana) si J1 usa umbral s y J2 responde de forma óptima.  g(s) = 1/4 + s/2 - s⁴/2."""
    s = np.asarray(s, dtype=float)
    return 0.25 + s / 2 - s**4 / 2


def prob_j2_tras_plantarse(s, t):
    """P(J2 gana | J1 se plantó o complementó) si J1 usa umbral s y J2 usa umbral t.

    Dado el anuncio, V1 ~ U(s, 1). Plantado con m, J2 gana con (m-s)⁺/(1-s);
    descartando gana con 1 - E[V1] = (1-s)/2.
    """
    return (2 * t - 1) * (1 - s) / 2 + ((1 - s) ** 2 - np.maximum(t - s, 0.0) ** 2) / (1 - s)


def mejor_respuesta_j1(t, tau=0.5):
    """Umbral óptimo de J1 si J2 usa umbrales (t, tau): menor m con pago_plantarse(m, t) >= pago_descartar(tau)."""
    r = pago_descartar(tau)
    if (2 * t - 1) / 2 >= r:          # hasta con M1 = 1/2 conviene plantarse
        return 0.5
    if (2 * t - 1) * t >= r:          # el cruce está por debajo de t
        return r / (2 * t - 1)
    return min((r + 2 * t) / (2 * t + 1), 1.0)


def iterar_mejores_respuestas(s0: float, pasos: int):
    """«Yo pienso que él piensa…»: s_{k+1} = mejor respuesta de J1 a la mejor respuesta de J2 a s_k.

    Devuelve las sucesiones (s_k) y (t_k = umbral_j2(s_k)). Cerca de s* la composición tiene
    derivada -1/(2 s*³) = -2, así que el punto fijo es inestable: la sucesión oscila.
    """
    s = [float(s0)]
    t = []
    for _ in range(pasos):
        t.append(float(umbral_j2(s[-1])))
        s.append(mejor_respuesta_j1(t[-1], 0.5))
    t.append(float(umbral_j2(s[-1])))
    return np.array(s), np.array(t)


# ---------------------------------------------------------------------------
# Variante A (modificación): el anuncio de J1 es secreto
# ---------------------------------------------------------------------------
UMBRAL_OCULTO = 1 / math.sqrt(2)
P_GANA_J1_OCULTO = 0.5


def prob_j1_oculto(s, t):
    """P(J1 gana) si nadie ve anuncios: J1 de umbral s, J2 de umbral único t (= prob_j1 con tau = t)."""
    return prob_j1(s, t, t)


def mejor_respuesta_oculto(t):
    """Umbral óptimo de un jugador si el rival (que no ve nada) usa umbral t."""
    return mejor_respuesta_j1(t, t)


# ---------------------------------------------------------------------------
# Variante B (reducción): no existe la opción «complemento»
# ---------------------------------------------------------------------------
def _cardano_sin_complemento() -> float:
    """Raíz real de s³ + s - 3/4 = 0 (fórmula de Cardano)."""
    q = -0.75
    d = math.sqrt(q * q / 4 + 1 / 27)
    return math.copysign(abs(-q / 2 + d) ** (1 / 3), -q / 2 + d) + math.copysign(abs(-q / 2 - d) ** (1 / 3), -q / 2 - d)


S_SIN_COMPLEMENTO = _cardano_sin_complemento()
T_SIN_COMPLEMENTO = (1 + S_SIN_COMPLEMENTO**2) / 2
TAU_SIN_COMPLEMENTO = 0.5


def prob_j1_sin_complemento(s, t, tau=0.5):
    """P(J1 gana) sin complemento: J1 se planta si x >= s; J2 se planta si z >= t (tras plantarse) o z >= tau (tras descartar)."""
    descartar = tau / 2 + (1 - tau) ** 2 / 2
    plantarse = t * (1 - s**2) / 2 + ((1 - t) ** 2 - np.maximum(s - t, 0.0) ** 2) / 2
    return s * descartar + plantarse


def prob_j1_umbral_sin_complemento(s):
    """h(s) = 3/8 + 3s/8 - s²/4 - s⁴/8: J1 de umbral s contra la mejor respuesta de J2 (sin complemento)."""
    s = np.asarray(s, dtype=float)
    return 3 / 8 + 3 * s / 8 - s**2 / 4 - s**4 / 8


P_GANA_J1_SIN_COMPLEMENTO = float(prob_j1_umbral_sin_complemento(S_SIN_COMPLEMENTO))


def resumen() -> dict:
    """Números clave del equilibrio, para reportes."""
    s, t = S_ESTRELLA, T_ESTRELLA
    return {
        "s_estrella": s,
        "t_estrella": t,
        "tau_estrella": TAU_ESTRELLA,
        "p_gana_j1": P_GANA_J1,
        "p_gana_j2": P_GANA_J2,
        "p_j1_planta": 2 * (1 - s),
        "p_j1_descarta": 2 * s - 1,
        "p_j2_descarta_si_j1_planto": s**2,
        "media_v1_si_planto": (1 + s) / 2,
        "p_gana_j1_si_planto": (1 - s**4) / (4 * (1 - s)),
        "p_gana_j1_si_descarto": 0.25,
        "p_gana_j1_ambos_plantados": (1 - t) / (1 - s) / 2,
        "p_gana_j1_planto_y_j2_descarto": (1 + s) / 2,
    }
