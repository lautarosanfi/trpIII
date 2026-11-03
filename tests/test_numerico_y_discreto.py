"""Mejor respuesta exacta para estrategias arbitrarias y juego discreto resuelto por PL."""

import numpy as np
import pytest

from juego import analitico as A
from juego import discreto as disc
from juego import modelo as M
from juego import numerico as Nu

S, T = A.S_ESTRELLA, A.T_ESTRELLA


@pytest.mark.parametrize("s", [0.55, S, 0.75])
def test_mejor_respuesta_exacta_a_umbral(s):
    _, u = Nu.mejor_respuesta_j2(M.j1_umbral(s))
    assert u[M.PLANTARSE] == pytest.approx(float(A.umbral_j2(s)), abs=2e-4)
    assert u[M.DESCARTAR] == pytest.approx(0.5)
    assert Nu.prob_j1_contra_mejor_respuesta(M.j1_umbral(s)) == pytest.approx(float(A.prob_j1_umbral(s)), abs=1e-5)


@pytest.mark.parametrize("e1", [M.j1_farol_aleatorio(S, 0.1), M.j1_farol_aleatorio(S, 0.5),
                                M.j1_farol_polarizado(S, 0.55), M.j1_umbrales_por_anuncio(0.6, 0.66),
                                M.j1_umbral(0.5)])
def test_ninguna_desviacion_de_j1_mejora(e1):
    """Contra σ2* (y con más razón contra la mejor respuesta) J1 no puede superar el valor del juego."""
    assert Nu.prob_j1_contra_umbrales(e1, T, T, 0.5) < A.P_GANA_J1 - 1e-4
    assert Nu.prob_j1_contra_mejor_respuesta(e1) <= Nu.prob_j1_contra_umbrales(e1, T, T, 0.5) + 1e-6


def test_umbrales_distintos_por_anuncio_promedian():
    v = Nu.prob_j1_contra_mejor_respuesta(M.j1_umbrales_por_anuncio(0.6, 0.7))
    assert v == pytest.approx(float((A.prob_j1_umbral(0.6) + A.prob_j1_umbral(0.7)) / 2), abs=1e-5)


def test_pl_original():
    sol = disc.resolver(160)
    assert sol.valor == pytest.approx(sol.valor_j2, abs=1e-8)       # minimax
    assert sol.valor == pytest.approx(A.P_GANA_J1, abs=2e-4)
    lo, hi = sol.region_descarte_j1()
    assert lo == pytest.approx(1 - S, abs=2 / 160) and hi == pytest.approx(S, abs=2 / 160)
    for h in (M.PLANTARSE, M.COMPLEMENTO):
        lo, hi = sol.region_descarte_j2(h)
        assert lo == pytest.approx(1 - T, abs=3 / 160) and hi == pytest.approx(T, abs=3 / 160)
    assert sol.region_descarte_j2(M.DESCARTAR) is None   # tras un descarte J2 nunca descarta
    assert sol.celdas_mixtas_j1() <= 2                    # no hay faroles: estrategia (casi) pura


@pytest.mark.parametrize("variante,valor", [("sin_complemento", A.P_GANA_J1_SIN_COMPLEMENTO),
                                            ("anuncio_oculto", 0.5)])
def test_pl_variantes(variante, valor):
    sol = disc.resolver(100, variante)
    assert sol.valor == pytest.approx(sol.valor_j2, abs=1e-8)
    assert sol.valor == pytest.approx(valor, abs=5e-4)


def test_juego_ficticio_converge():
    fp = disc.juego_ficticio(40, 3000, registrar=20)
    assert fp["brecha"][-1] < 2e-3
    assert fp["brecha"][-1] < fp["brecha"][0] / 100
    assert fp["valor_inf"][-1] <= A.P_GANA_J1 + 2e-3 <= fp["valor_sup"][-1] + 4e-3


def test_matriz_pago_discreta_reproduce_las_formulas_continuas():
    n = 50
    x = disc.grilla(n)
    U = disc.matriz_pago(M.PLANTARSE, M.DESCARTAR, n)
    assert np.allclose(U[:, 0], x)              # P(W < x) = x
    U = disc.matriz_pago(M.DESCARTAR, M.COMPLEMENTO, n)
    assert np.allclose(U[0, :], 1 - (1 - x))     # J2 con 1-z: P(Y > 1-z) = z
    U = disc.matriz_pago(M.COMPLEMENTO, M.PLANTARSE, n)
    assert U[0, n - 1] == 0.5                   # empate exacto: 1 - x_0 = x_{n-1}
