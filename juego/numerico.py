"""Mejor respuesta de J2 a una estrategia **cualquiera** de J1.

Dos maneras de obtenerla, que deben coincidir:

1. «Exacta» (integración numérica): se discretiza x en K celdas muy finas, se arma la
   distribución de V1 tras cada anuncio (regla de Bayes) y se calcula la respuesta de J2.
2. «Empírica» (aprendizaje): J2 observa muchas partidas simuladas — el anuncio de J1 y,
   al final, su número — y estima esas mismas distribuciones con los datos.

En ambos casos la regla de decisión de J2 es la misma (Lema 1 del entregable): con
M2 = max(z, 1-z), plantarse gana con P(V1 < M2 | anuncio) y descartar gana con
1 - E[V1 | anuncio]. Por lo tanto J2 se planta si y sólo si M2 supera el cuantil de
orden 1 - E[V1 | anuncio] de la distribución de V1 dado el anuncio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analitico import pago_descartar, pago_plantarse
from .modelo import COMPLEMENTO, DESCARTAR, PLANTARSE, EstrategiaJ1, EstrategiaJ2, j2_umbrales

K_DEFECTO = 20_000


@dataclass
class Posterior:
    """Lo que J2 sabe del número final de J1 después de un anuncio."""

    prob_anuncio: float     # P(J1 hace este anuncio)
    valores: np.ndarray     # valores posibles de V1, ordenados
    acumulada: np.ndarray   # P(V1 <= valores[k] | anuncio)
    media: float            # E[V1 | anuncio]

    def cdf(self, m):
        k = np.searchsorted(self.valores, m, side="right")
        return np.where(k > 0, self.acumulada[np.maximum(k - 1, 0)], 0.0)

    def umbral_optimo(self) -> float:
        """Menor M2 en [1/2, 1] con P(V1 <= M2) >= 1 - E[V1]: el umbral de la mejor respuesta de J2."""
        k = np.searchsorted(self.acumulada, 1 - self.media - 1e-12, side="left")
        return float(max(0.5, self.valores[min(k, len(self.valores) - 1)]))

    def prob_gana_j2(self, k_integracion: int = K_DEFECTO) -> float:
        """P(J2 gana | anuncio) si J2 responde óptimamente: E[max(F(M2), 1 - E[V1])], M2 ~ U(1/2,1)."""
        m = 0.5 + (np.arange(k_integracion) + 0.5) / (2 * k_integracion)
        return float(np.mean(np.maximum(self.cdf(m), 1 - self.media)))


def _posterior_discreta(valores, pesos) -> Posterior | None:
    total = float(pesos.sum())
    if total <= 0:
        return None
    orden = np.argsort(valores, kind="stable")
    v, w = valores[orden], pesos[orden] / total
    return Posterior(total, v, np.cumsum(w), float(np.dot(v, w)))


def posteriores(e1: EstrategiaJ1, k: int = K_DEFECTO) -> dict:
    """Distribución de V1 tras cada anuncio, por regla de Bayes sobre una grilla de k celdas de x.

    Tras «descartar» V1 = Y es un U(0,1) nuevo, independiente de x: la posterior es U(0,1)
    sin importar la estrategia de J1 (se representa con la misma grilla).
    """
    x = (np.arange(k) + 0.5) / k
    p = e1.probs(x)
    grilla = (np.arange(k) + 0.5) / k
    post = {
        PLANTARSE: _posterior_discreta(x, p[:, PLANTARSE] / k),
        COMPLEMENTO: _posterior_discreta(1 - x, p[:, COMPLEMENTO] / k),
        DESCARTAR: None,
    }
    prob_d = float(p[:, DESCARTAR].mean())
    if prob_d > 0:
        post[DESCARTAR] = Posterior(prob_d, grilla, (np.arange(k) + 1) / k, 0.5)
    return post


def mejor_respuesta_j2(e1: EstrategiaJ1, k: int = K_DEFECTO) -> tuple[EstrategiaJ2, dict]:
    """Mejor respuesta de J2 (umbral por anuncio) a la estrategia e1. Anuncios que nunca ocurren:
    se usa la creencia «V1 ~ U(0,1)» (umbral 1/2); para «descartar» es la única creencia posible."""
    post = posteriores(e1, k)
    umbrales = {a: (post[a].umbral_optimo() if post[a] is not None else 0.5)
                for a in (PLANTARSE, COMPLEMENTO, DESCARTAR)}
    e2 = j2_umbrales(umbrales[PLANTARSE], umbrales[DESCARTAR], umbrales[COMPLEMENTO],
                     nombre="J2 mejor respuesta (exacta)")
    return e2, umbrales


def prob_j1_contra_mejor_respuesta(e1: EstrategiaJ1, k: int = K_DEFECTO) -> float:
    """P(J1 gana) si J2 conoce la estrategia e1 y le responde de forma óptima."""
    post = posteriores(e1, k)
    gana_j2 = sum(pa.prob_anuncio * pa.prob_gana_j2(k) for pa in post.values() if pa is not None)
    return 1.0 - gana_j2


def prob_j1_contra_umbrales(e1: EstrategiaJ1, t_p: float, t_c: float, t_d: float,
                            k: int = K_DEFECTO) -> float:
    """P(J1 gana) si J1 juega e1 y J2 usa umbrales fijos (t_p, t_c, t_d) según el anuncio."""
    x = (np.arange(k) + 0.5) / k
    p = e1.probs(x)
    pago = (p[:, PLANTARSE] * pago_plantarse(x, t_p)
            + p[:, COMPLEMENTO] * pago_plantarse(1 - x, t_c)
            + p[:, DESCARTAR] * pago_descartar(t_d))
    return float(pago.mean())


# ---------------------------------------------------------------------------
# Aprendizaje a partir de partidas simuladas
# ---------------------------------------------------------------------------
def umbrales_empiricos(anuncios, numeros) -> dict:
    """Umbral óptimo de J2 para cada anuncio, estimado con datos (anuncio de J1, número final de J1).

    Regla: t = max(1/2, cuantil de orden 1 - media de los números observados tras ese anuncio).
    """
    umbrales = {}
    for a in (PLANTARSE, COMPLEMENTO, DESCARTAR):
        v = numeros[anuncios == a]
        if len(v) == 0:
            umbrales[a] = 0.5
            continue
        nivel = 1 - v.mean()
        umbrales[a] = float(max(0.5, np.quantile(v, nivel, method="inverted_cdf")))
    return umbrales


def mejor_respuesta_empirica(anuncios, numeros) -> tuple[EstrategiaJ2, dict]:
    u = umbrales_empiricos(anuncios, numeros)
    e2 = j2_umbrales(u[PLANTARSE], u[DESCARTAR], u[COMPLEMENTO], nombre="J2 aprendida de datos")
    return e2, u
