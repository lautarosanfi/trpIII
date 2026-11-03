"""El simulador (que sigue las reglas al pie de la letra) contra las fórmulas."""

import numpy as np
import pytest

from juego import analitico as A
from juego import modelo as M
from juego import numerico as Nu

S, T = A.S_ESTRELLA, A.T_ESTRELLA


def test_numero_final():
    acc = np.array([M.PLANTARSE, M.COMPLEMENTO, M.DESCARTAR])
    out = M.numero_final(acc, np.array([0.2, 0.2, 0.2]), np.array([0.9, 0.9, 0.9]))
    assert out.tolist() == pytest.approx([0.2, 0.8, 0.9])


def test_muestrear():
    rng = np.random.default_rng(0)
    puras = np.eye(3)[[0, 1, 2, 2, 1]]
    assert M.muestrear(puras, rng).tolist() == [0, 1, 2, 2, 1]
    mixta = np.tile([0.2, 0.3, 0.5], (200_000, 1))
    frec = np.bincount(M.muestrear(mixta, rng), minlength=3) / 200_000
    assert frec == pytest.approx([0.2, 0.3, 0.5], abs=0.005)


def test_equilibrio_por_montecarlo():
    e = M.estimar_prob_j1(M.j1_umbral(S), M.j2_umbrales(T, 0.5), 1_000_000, np.random.default_rng(1))
    assert abs(e.p - A.P_GANA_J1) < 4 * e.error_estandar


@pytest.mark.parametrize("s,t,tau", [(0.5, 0.5, 0.5), (0.6, 0.75, 0.55), (0.72, 0.62, 0.8), (0.9, 0.95, 0.5)])
def test_formula_general_por_montecarlo(s, t, tau):
    e = M.estimar_prob_j1(M.j1_umbral(s), M.j2_umbrales(t, tau), 600_000, np.random.default_rng(2))
    assert abs(e.p - float(A.prob_j1(s, t, tau))) < 4 * e.error_estandar


def test_sin_complemento_por_montecarlo():
    s, t = A.S_SIN_COMPLEMENTO, A.T_SIN_COMPLEMENTO
    e = M.estimar_prob_j1(M.j1_sin_complemento(s), M.j2_sin_complemento(t, 0.5), 600_000,
                          np.random.default_rng(3))
    assert abs(e.p - A.P_GANA_J1_SIN_COMPLEMENTO) < 4 * e.error_estandar


def test_anuncio_oculto_por_montecarlo():
    u = A.UMBRAL_OCULTO
    e = M.estimar_prob_j1(M.j1_umbral(u), M.j2_ignora_anuncio(u), 600_000, np.random.default_rng(4),
                          ve_anuncio=False)
    assert abs(e.p - 0.5) < 4 * e.error_estandar


def test_j2_no_puede_espiar_si_el_anuncio_es_secreto():
    with pytest.raises(ValueError):
        M.simular(M.j1_umbral(S), M.j2_umbrales(T, 0.5), 10, np.random.default_rng(5), ve_anuncio=False)


@pytest.mark.parametrize("s", [0.5, 0.6, S, 0.75])
def test_j2_aprende_los_umbrales_de_los_datos(s):
    a1, v1 = M.simular_j1(M.j1_umbral(s), 400_000, np.random.default_rng(6))
    u = Nu.umbrales_empiricos(a1, v1)
    assert u[M.PLANTARSE] == pytest.approx(float(A.umbral_j2(s)), abs=0.006)
    assert u[M.COMPLEMENTO] == pytest.approx(float(A.umbral_j2(s)), abs=0.006)
    assert u[M.DESCARTAR] == pytest.approx(0.5, abs=0.006)
