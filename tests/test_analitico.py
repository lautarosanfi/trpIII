"""Consistencia interna de las fórmulas cerradas."""

import math

import numpy as np
import pytest

from juego import analitico as A

S, T = A.S_ESTRELLA, A.T_ESTRELLA


def test_umbrales_del_equilibrio():
    assert S**3 == pytest.approx(0.25)
    assert T == pytest.approx((1 + S**2) / 2)
    assert A.P_GANA_J1 == pytest.approx(0.25 + 3 / (8 * 4 ** (1 / 3)))
    assert A.P_GANA_J1 + A.P_GANA_J2 == pytest.approx(1)


def test_g_es_maxima_en_s_estrella():
    s = np.linspace(0.5, 1, 2001)
    g = A.prob_j1_umbral(s)
    assert s[np.argmax(g)] == pytest.approx(S, abs=1e-3)
    h = 1e-6
    assert (A.prob_j1_umbral(S + h) - A.prob_j1_umbral(S - h)) / (2 * h) == pytest.approx(0, abs=1e-6)
    assert float(A.prob_j1_umbral(0.5)) == pytest.approx(15 / 32)


@pytest.mark.parametrize("s", [0.5, 0.55, 0.6, S, 0.7, 0.85, 0.99])
def test_formula_general_coincide_con_g(s):
    assert float(A.prob_j1(s, A.umbral_j2(s), 0.5)) == pytest.approx(float(A.prob_j1_umbral(s)))


def test_mejores_respuestas_se_cruzan_en_el_equilibrio():
    assert A.mejor_respuesta_j1(T, 0.5) == pytest.approx(S)
    assert A.pago_plantarse(S, T) == pytest.approx(A.pago_descartar(0.5))  # indiferencia de J1
    assert A.pago_descartar(0.5) == pytest.approx(0.25)


def test_punto_fijo_inestable():
    h = 1e-6
    phi = lambda s: A.mejor_respuesta_j1(float(A.umbral_j2(s)))  # noqa: E731
    derivada = (phi(S + h) - phi(S - h)) / (2 * h)
    assert derivada == pytest.approx(-2, rel=1e-4)
    s_seq, _ = A.iterar_mejores_respuestas(0.5, 30)
    assert abs(s_seq[-1] - S) > 0.05  # no converge


def test_tabla_resumen():
    r = A.resumen()
    total = r["p_j1_planta"] * r["p_gana_j1_si_planto"] + r["p_j1_descarta"] * r["p_gana_j1_si_descarto"]
    assert total == pytest.approx(A.P_GANA_J1)
    pl = (1 - r["p_j2_descarta_si_j1_planto"]) * r["p_gana_j1_ambos_plantados"] \
        + r["p_j2_descarta_si_j1_planto"] * r["p_gana_j1_planto_y_j2_descarto"]
    assert pl == pytest.approx(r["p_gana_j1_si_planto"])


def test_variante_anuncio_oculto():
    u = A.UMBRAL_OCULTO
    assert A.mejor_respuesta_oculto(u) == pytest.approx(u)
    assert float(A.prob_j1_oculto(u, u)) == pytest.approx(0.5)
    assert u == pytest.approx(1 / math.sqrt(2))


def test_variante_sin_complemento():
    s, t = A.S_SIN_COMPLEMENTO, A.T_SIN_COMPLEMENTO
    assert s**3 + s == pytest.approx(0.75)
    grilla = np.linspace(0.3, 1, 7001)
    assert grilla[np.argmax(A.prob_j1_umbral_sin_complemento(grilla))] == pytest.approx(s, abs=1e-3)
    assert float(A.prob_j1_sin_complemento(s, t, 0.5)) == pytest.approx(A.P_GANA_J1_SIN_COMPLEMENTO)
    # indiferencia de J1 en s: plantarse (t·s) vale lo mismo que descartar (3/8)
    assert t * s == pytest.approx(3 / 8)
