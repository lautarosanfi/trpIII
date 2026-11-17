#!/usr/bin/env python3
"""Genera las figuras (SVG) a partir de resultados/resultados.json.

Uso (desde la raíz del repo, después de scripts/explorar.py):
    python scripts/graficar.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from juego import analitico as A  # noqa: E402
from juego import figuras as F  # noqa: E402
from juego.figuras import (ANCHO, ANCHO_DOBLE, AZUL, COLOR_ACCION, GRIS, LADRILLO, PAPEL, REGLA,  # noqa: E402
                           TINTA, TINTA2, VERDE, coma, etiqueta, formato_coma, nota, puntos)

S, T = A.S_ESTRELLA, A.T_ESTRELLA
CARPETA = RAIZ / "resultados" / "figuras"


def fig_distribucion_m(R):
    e0 = R["e0"]
    b = np.array(e0["bordes"])
    centros, w = (b[:-1] + b[1:]) / 2, (b[1] - b[0]) * 0.86
    fig, ax = plt.subplots(figsize=(ANCHO, 2.3))
    ax.bar(centros, e0["densidad_x"], width=w, color=REGLA, lw=0)
    ax.bar(centros, e0["densidad_m"], width=w, color=AZUL, alpha=0.85, lw=0)
    ax.plot([0, 1], [1, 1], color=TINTA2, lw=0.8)
    ax.plot([0.5, 1], [2, 2], color=AZUL, lw=0.8)
    etiqueta(ax, 0.02, 1.13, "x es uniforme en (0, 1): densidad 1", ha="left", va="bottom")
    etiqueta(ax, 1.0, 2.08, "M = máx(x, 1 − x) es uniforme en (½, 1): densidad 2", ha="right", va="bottom",
             color=TINTA)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2.45)
    ax.set_yticks([0, 1, 2])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    formato_coma(ax, 2, 0)
    ax.set_xlabel("valor")
    ax.set_ylabel("densidad")
    F.guardar(fig, CARPETA, "distribucion_m")


def fig_mc_convergencia(R):
    tr = R["e9"]["trayectoria"]
    n, p = np.array(tr["n"]), np.array(tr["p"])
    n, p = n[n >= 100], p[n >= 100]
    fig, ax = plt.subplots(figsize=(ANCHO, 2.5))
    nn = np.geomspace(n.min(), n.max(), 300)
    ee = 1.96 * np.sqrt(A.P_GANA_J1 * (1 - A.P_GANA_J1) / nn)
    ax.fill_between(nn, A.P_GANA_J1 - ee, A.P_GANA_J1 + ee, color=AZUL, alpha=0.09, lw=0)
    ax.axhline(A.P_GANA_J1, color=TINTA2, lw=0.7)
    ax.plot(n, p, color=AZUL, lw=1.2)
    etiqueta(ax, 130, A.P_GANA_J1 + 0.073, "banda de ±1,96 errores estándar", va="bottom")
    etiqueta(ax, n[-1], A.P_GANA_J1 - 0.012, f"valor teórico {coma(A.P_GANA_J1, 4)}", ha="right", va="top")
    ax.set_xscale("log")
    ax.set_ylim(0.40, 0.575)
    ax.set_xlabel("cantidad de partidas simuladas")
    ax.set_ylabel("proporción de victorias de J1")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: coma(v, 2)))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", ".")))
    ax.minorticks_off()
    F.guardar(fig, CARPETA, "mc_convergencia")


def fig_g_de_s(R):
    filas = R["e2"]["filas"]
    s = np.array([r["s"] for r in filas])
    p = np.array([r["prob"]["p"] for r in filas])
    ee = np.array([r["prob"]["ee"] for r in filas])
    ss = np.linspace(0.5, 0.8, 400)
    fig, ax = plt.subplots(figsize=(ANCHO, 3.1))
    ax.axhline(0.5, color=REGLA, lw=0.8)
    etiqueta(ax, 0.797, 0.5012, "½: juego parejo", ha="right", va="bottom", color=GRIS)
    ax.plot(ss, A.prob_j1_umbral(ss), color=AZUL)
    ax.errorbar(s, p, yerr=1.96 * ee, fmt="none", ecolor=GRIS, elinewidth=0.6, capsize=0, zorder=3)
    puntos(ax, s, p, AZUL)
    etiqueta(ax, 0.752, 0.4762, "fórmula g(s)", color=AZUL)
    etiqueta(ax, 0.705, 0.4892, "simulación", color=TINTA2)
    for x0, txt, xt in [(0.5, "nunca descartar\n15/32 ≈ 0,469", (0.515, 0.452)),
                        (S, f"máximo en s* ≈ {coma(S, 3)}\nP = {coma(A.P_GANA_J1, 4)}", (0.585, 0.4965)),
                        (1 / np.sqrt(2), "umbral 1/√2\n(el óptimo si el anuncio\nfuera secreto)", (0.6, 0.452))]:
        y0 = float(A.prob_j1_umbral(x0))
        ax.plot([x0], [y0], "o", ms=5.2, color=TINTA, mec=PAPEL, mew=1.2, zorder=5)
        nota(ax, txt, (x0, y0), xt)
    ax.set_xlim(0.495, 0.80)
    ax.set_ylim(0.44, 0.505)
    ax.set_xlabel("umbral s de J1 (se queda con máx(x, 1 − x) si es ≥ s; si no, descarta)")
    ax.set_ylabel("P(J1 gana)")
    formato_coma(ax, 2, 2)
    F.guardar(fig, CARPETA, "g_de_s")


def fig_umbrales_j2(R):
    filas = R["e2"]["filas"]
    s = np.array([r["s"] for r in filas])
    up = np.array([(r["umbral_p"] + r["umbral_c"]) / 2 for r in filas])
    ud = np.array([r["umbral_d"] for r in filas])
    ss = np.linspace(0.5, 0.8, 300)
    fig, ax = plt.subplots(figsize=(ANCHO, 3.0))
    ax.plot(ss, (1 + ss) / 2, color=GRIS, lw=1, ls=(0, (4, 3)))
    ax.plot(ss, A.umbral_j2(ss), color=AZUL)
    puntos(ax, s[::2], up[::2], AZUL)
    ax.plot(ss, np.full_like(ss, 0.5), color=LADRILLO)
    puntos(ax, s[::2], ud[::2], LADRILLO)
    etiqueta(ax, 0.802, 0.9, "promedio de J1 si se plantó, (1 + s)/2", va="center", color=GRIS)
    etiqueta(ax, 0.802, float(A.umbral_j2(0.8)), "umbral de J2 tras «me planto»,\n(1 + s²)/2", va="center",
             color=AZUL)
    etiqueta(ax, 0.802, 0.5, "umbral de J2 tras «descarto»: ½", va="center", color=LADRILLO)
    ax.axvline(S, color=REGLA, lw=0.8, zorder=0)
    nota(ax, f"con s = s*, J2 se queda desde {coma(T, 3)}\naunque J1 promedie {coma((1 + S) / 2, 3)}",
         (S, T), (0.66, 0.6))
    ax.set_xlim(0.495, 0.80)
    ax.set_ylim(0.47, 0.93)
    ax.set_xlabel("umbral s de J1")
    ax.set_ylabel("umbral de J2 sobre máx(z, 1 − z)")
    formato_coma(ax, 2, 2)
    F.guardar(fig, CARPETA, "umbrales_j2")


def fig_decision_j2(R):
    m = np.linspace(0.5, 1, 400)
    gana_pl = np.clip((m - S) / (1 - S), 0, 1)
    gana_d = (1 - S) / 2
    fig, ax = plt.subplots(figsize=(ANCHO, 2.9))
    for x0, txt in [(0.625, "0,625"), (0.75, "0,75"), ((1 + S) / 2, "promedio\nde J1: 0,815")]:
        ax.axvline(x0, color=REGLA, lw=0.8, zorder=0)
        etiqueta(ax, x0, 1.015, txt, ha="center", va="bottom", color=GRIS, tam=8.2)
    ax.plot(m, gana_pl, color=AZUL)
    ax.plot(m, np.full_like(m, gana_d), color=LADRILLO)
    etiqueta(ax, 0.905, 0.79, "quedarse con m:\nP(V₁ < m | J1 se plantó)", color=AZUL, ha="right")
    etiqueta(ax, 0.51, gana_d + 0.03, "descartar: 1 − E[V₁ | J1 se plantó]", color=LADRILLO, va="bottom")
    ax.plot([T], [gana_d], "o", ms=5.5, color=TINTA, mec=PAPEL, mew=1.2, zorder=5)
    nota(ax, f"se cruzan en t* ≈ {coma(T, 3)}", (T, gana_d), (0.76, 0.08))
    ax.set_xlim(0.5, 1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("m = máx(z, 1 − z): lo mejor que J2 puede quedarse sin descartar")
    ax.set_ylabel("P(J2 gana)")
    formato_coma(ax, 2, 1)
    F.guardar(fig, CARPETA, "decision_j2")


def fig_decision_j1(R):
    m = np.linspace(0.5, 1, 400)
    fig, ax = plt.subplots(figsize=(ANCHO, 2.9))
    ax.plot(m, A.pago_plantarse(m, T), color=AZUL)
    ax.plot(m, np.full_like(m, 0.25), color=LADRILLO)
    etiqueta(ax, 0.86, 0.80, "quedarse con m", color=AZUL, ha="right")
    etiqueta(ax, 0.995, 0.28, "descartar: ¼", color=LADRILLO, ha="right", va="bottom")
    ax.plot([S], [0.25], "o", ms=5.5, color=TINTA, mec=PAPEL, mew=1.2, zorder=5)
    nota(ax, f"s* ≈ {coma(S, 3)}: s*·s*² = ¼", (S, 0.25), (0.665, 0.085))
    for x0 in (0.52, 0.55):
        puntos(ax, [x0], [float(A.pago_plantarse(x0, T))], AZUL, tam=5)
    nota(ax, f"con 0,52 gana {coma(float(A.pago_plantarse(0.52, T)), 3)}\ncon 0,55 gana "
             f"{coma(float(A.pago_plantarse(0.55, T)), 3)}", (0.52, float(A.pago_plantarse(0.52, T))), (0.535, 0.47))
    nota(ax, "desde t* también le gana\na un J2 que se plantó", (T, float(A.pago_plantarse(T, T))), (0.775, 0.36))
    ax.set_xlim(0.5, 1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("m = máx(x, 1 − x)")
    ax.set_ylabel("P(J1 gana)")
    formato_coma(ax, 2, 1)
    F.guardar(fig, CARPETA, "decision_j1")


def fig_faroles(R):
    e5 = R["e5"]
    fig, axs = plt.subplots(1, 2, figsize=(ANCHO_DOBLE, 2.9), sharey=True)
    for ax, clave, xlab, tit in [
        (axs[0], "aleatorio", "q: probabilidad de plantarse igual si máx(x, 1 − x) < s*", "a) farol al azar"),
        (axs[1], "polarizado", "b: también se planta si máx(x, 1 − x) < b", "b) farol con los peores números"),
    ]:
        filas = e5[clave]
        xv = np.array([r["q" if clave == "aleatorio" else "b"] for r in filas])
        ax.axhline(A.P_GANA_J1, color=REGLA, lw=0.8)
        ax.plot(xv, [r["vs_t_estrella"] for r in filas], color=GRIS, lw=1.3)
        ax.plot(xv, [r["vs_mejor_respuesta"] for r in filas], color=TINTA, lw=1.4)
        ax.set_title(tit, fontstyle="italic")
        ax.set_xlabel(xlab)
        formato_coma(ax, 1 if clave == "aleatorio" else 2, 3)
    a = axs[0]
    etiqueta(a, 0.99, A.P_GANA_J1 + 0.0005, "sin farol (equilibrio)", ha="right", va="bottom", color=GRIS)
    etiqueta(a, 0.6, 0.4831, "J2 no lo anticipa", color=GRIS, va="bottom")
    etiqueta(a, 0.55, 0.4745, "J2 se adapta", color=TINTA, va="top")
    mc = e5["mc_farol_03"]["mc"]
    a.errorbar([0.3], [mc["p"]], yerr=[1.96 * mc["ee"]], fmt="none", ecolor=TINTA2, elinewidth=0.7)
    puntos(a, [0.3], [mc["p"]], AZUL, tam=5)
    nota(a, "simulación", (0.3, mc["p"]), (0.12, 0.4795))
    etiqueta(axs[1], 0.6, 0.4812, "no lo anticipa", color=GRIS, va="bottom")
    etiqueta(axs[1], 0.565, 0.4728, "se adapta", color=TINTA, va="top")
    axs[0].set_ylabel("P(J1 gana)")
    axs[0].set_ylim(0.466, 0.4885)
    axs[0].yaxis.set_major_locator(plt.MultipleLocator(0.005))
    axs[1].set_xlim(0.5, S)
    axs[1].xaxis.set_major_locator(plt.MultipleLocator(0.05))
    fig.subplots_adjust(wspace=0.1)
    F.guardar(fig, CARPETA, "faroles")


def fig_telarana(R):
    s_seq = np.array(R["e6"]["s"])
    fig, axs = plt.subplots(1, 2, figsize=(ANCHO_DOBLE, 3.1), gridspec_kw={"width_ratios": [1, 1.45]})
    ax = axs[0]
    ss = np.linspace(0.5, 0.75, 400)
    phi = np.array([A.mejor_respuesta_j1(A.umbral_j2(x)) for x in ss])
    ax.plot(ss, ss, color=REGLA, lw=0.9)
    ax.plot(ss, phi, color=AZUL)
    xs, ys = [s_seq[0]], [s_seq[0]]
    for k in range(10):
        xs += [s_seq[k], s_seq[k + 1]]
        ys += [s_seq[k + 1], s_seq[k + 1]]
    ax.plot(xs, ys, color=LADRILLO, lw=0.9)
    ax.plot([S], [S], "o", ms=5.2, color=TINTA, mec=PAPEL, mew=1.2, zorder=5)
    nota(ax, "punto fijo s*", (S, S), (0.515, 0.72))
    etiqueta(ax, 0.715, 0.53, "φ(s)", color=AZUL)
    etiqueta(ax, 0.745, 0.728, "diagonal", color=GRIS, ha="right", va="top")
    ax.set_xlim(0.5, 0.75)
    ax.set_ylim(0.5, 0.75)
    ax.set_aspect("equal")
    ax.set_title("a) diagrama de telaraña", fontstyle="italic")
    ax.set_xlabel("$s_k$")
    ax.set_ylabel("$s_{k+1}$")
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    formato_coma(ax, 2, 2)
    ax = axs[1]
    k = np.arange(len(s_seq))
    ax.axhline(S, color=REGLA, lw=0.8)
    etiqueta(ax, len(k) - 0.5, S + 0.004, "s*", ha="right", va="bottom", color=TINTA2)
    ax.plot(k, s_seq, color=LADRILLO, lw=0.9)
    puntos(ax, k, s_seq, LADRILLO, tam=3.6)
    ax.set_title("b) el umbral de J1 nunca se asienta", fontstyle="italic")
    ax.set_xlabel("ronda k")
    ax.set_ylabel("umbral de J1")
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: coma(v, 2)))
    fig.subplots_adjust(wspace=0.32)
    F.guardar(fig, CARPETA, "telarana")


def _franja(ax, x, probs, umbrales=(), etiquetas=True):
    """Probabilidad de cada acción (apiladas) en función del número, dibujada por tramos constantes."""
    probs = np.round(np.asarray(probs), 4)
    ancho_celda = x[1] - x[0]
    cambios = np.flatnonzero(np.any(np.diff(probs, axis=0) != 0, axis=1)) + 1
    for tramo in np.split(np.arange(len(x)), cambios):
        x0 = x[tramo[0]] - ancho_celda / 2
        w = len(tramo) * ancho_celda
        abajo = 0.0
        for a in (1, 2, 0):  # complemento, descartar, plantarse (de abajo hacia arriba)
            alto = probs[tramo[0], a]
            if alto > 0:
                # un leve solapamiento con el tramo siguiente evita «costuras» de antialiasing
                ax.add_patch(plt.Rectangle((x0, abajo), w + 0.6 * ancho_celda, alto, color=COLOR_ACCION[a], lw=0))
            abajo += alto
    for u in umbrales:
        ax.axvline(u, color=TINTA, lw=0.7, ls=(0, (2, 2)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for lado in ("left", "top", "right"):
        ax.spines[lado].set_visible(False)
    if etiquetas:
        accion = probs.argmax(axis=1)
        cortes = np.flatnonzero(np.diff(accion)) + 1
        for seg in np.split(np.arange(len(x)), cortes):
            a = accion[seg[0]]
            if x[seg[-1]] - x[seg[0]] > 0.12:
                ax.text(x[seg].mean(), 0.5, F.NOMBRE_ACCION[a], ha="center", va="center", fontsize=8.8,
                        color="white", fontstyle="italic")


def fig_pl_estrategias(R):
    e7 = R["e7"]
    n = e7["n"]
    x = (np.arange(n) + 0.5) / n
    r = np.array(e7["estrategia_j1"])
    q = np.array(e7["estrategia_j2"])
    fig, axs = plt.subplots(4, 1, figsize=(ANCHO, 3.5), sharex=True, gridspec_kw={"hspace": 0.95})
    filas = [(r, (1 - S, S), "J1, según su número x"),
             (q[0], (1 - T, T), "J2 si J1 anunció «me planto», según su número z"),
             (q[1], (1 - T, T), "J2 si J1 anunció «complemento»"),
             (q[2], (0.5,), "J2 si J1 anunció «descarto»")]
    for ax, (pr, umb, tit) in zip(axs, filas):
        _franja(ax, x, pr, umb)
        ax.set_title(tit, fontsize=8.8, color=TINTA2, pad=3, fontstyle="italic")
    axs[-1].set_xticks([0, 1 - T, 1 - S, 0.5, S, T, 1])
    axs[-1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: coma(v, 3) if 0 < v < 1 else coma(v, 0)))
    axs[-1].tick_params(axis="x", labelsize=7.8)
    F.guardar(fig, CARPETA, "pl_estrategias")


def fig_pl_convergencia(R):
    conv = R["e7"]["convergencia"]
    n = np.array([c["n"] for c in conv])
    fig, axs = plt.subplots(1, 2, figsize=(ANCHO_DOBLE, 2.7))
    ax = axs[0]
    ax.axhline(A.P_GANA_J1, color=REGLA, lw=0.8)
    etiqueta(ax, n[-1], A.P_GANA_J1 - 0.000008, f"juego continuo: {coma(A.P_GANA_J1, 5)}", ha="right", va="top",
             color=GRIS)
    ax.plot(n, [c["valor"] for c in conv], color=AZUL)
    puntos(ax, n, [c["valor"] for c in conv], AZUL, tam=4.6)
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([str(v) for v in n])
    ax.minorticks_off()
    ax.set_title("a) valor del juego discreto", fontstyle="italic")
    ax.set_xlabel("N (puntos de la grilla)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: coma(v, 5)))
    ax = axs[1]
    for y0, txt in ((S, f"s* ≈ {coma(S, 3)}"), (T, f"t* ≈ {coma(T, 3)}")):
        ax.axhline(y0, color=REGLA, lw=0.8)
        etiqueta(ax, n[-1] * 1.12, y0, txt, va="center", color=GRIS)
    ax.plot(n, [c["descarte_j1"][1] for c in conv], color=AZUL)
    puntos(ax, n, [c["descarte_j1"][1] for c in conv], AZUL, tam=4.6)
    ax.plot(n, [c["descarte_j2_tras_p"][1] for c in conv], color=LADRILLO)
    puntos(ax, n, [c["descarte_j2_tras_p"][1] for c in conv], LADRILLO, tam=4.6)
    etiqueta(ax, 25, 0.588, "mayor x que J1 descarta", color=AZUL, va="top")
    etiqueta(ax, 10, 0.712, "mayor z que J2 descarta tras «me planto»", color=LADRILLO, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([str(v) for v in n])
    ax.minorticks_off()
    ax.set_ylim(0.53, 0.75)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.set_title("b) umbrales", fontstyle="italic")
    ax.set_xlabel("N (puntos de la grilla)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: coma(v, 2)))
    fig.subplots_adjust(wspace=0.38)
    F.guardar(fig, CARPETA, "pl_convergencia")


def fig_juego_ficticio(R):
    e8 = R["e8"]
    fig, axs = plt.subplots(1, 2, figsize=(ANCHO_DOBLE, 2.8), gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axs[0]
    k, b = np.array(e8["iteracion"]), np.array(e8["brecha"])
    ax.plot(k, b, color=AZUL)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_title("a) brecha de explotabilidad", fontstyle="italic")
    ax.set_xlabel("ronda")
    ax.set_ylabel("brecha")
    ax = axs[1]
    r = np.array(e8["estrategia_j1"])
    x = (np.arange(len(r)) + 0.5) / len(r)
    _franja(ax, x, r, (1 - S, S))
    ax.set_title(f"b) estrategia promedio de J1 tras {k[-1]:,} rondas".replace(",", "."), fontstyle="italic")
    ax.set_xlabel("x (número de J1)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: coma(v, 1)))
    fig.subplots_adjust(wspace=0.22)
    F.guardar(fig, CARPETA, "juego_ficticio")


def fig_mapa_silla(R):
    """P(J1 gana) para cada par de umbrales (s de J1, t de J2 tras «me planto»), con τ = ½."""
    s = np.linspace(0.5, 1, 301)
    t = np.linspace(0.5, 1, 301)
    SS, TT = np.meshgrid(s, t)
    Z = A.prob_j1(SS, TT, 0.5)
    cmap = LinearSegmentedColormap.from_list(
        "div", [(0, "#8f3d22"), (0.25, "#d9a28b"), (0.5, "#f3efe5"), (0.75, "#a5b8dc"), (1, "#24467f")])
    fig, ax = plt.subplots(figsize=(ANCHO, 4.6))
    im = ax.imshow(Z, origin="lower", extent=(0.5, 1, 0.5, 1), cmap=cmap, vmin=0.25, vmax=0.75,
                   interpolation="bilinear", aspect="equal")
    niveles = np.arange(0.30, 0.71, 0.05)
    cs = ax.contour(SS, TT, Z, levels=niveles, colors=TINTA2, linewidths=0.45, alpha=0.7)
    ax.clabel(cs, fmt=lambda v: coma(v, 2), fontsize=7.4, inline=True, inline_spacing=3)
    ax.plot(s, A.umbral_j2(s), color=TINTA, lw=1.5)
    tt = np.linspace(0.5, 1, 601)
    ax.plot([A.mejor_respuesta_j1(v) for v in tt], tt, color=TINTA, lw=1.5, ls=(0, (4, 2.5)))
    ax.plot([S], [T], "o", ms=6.5, color=TINTA, mec=PAPEL, mew=1.4, zorder=6)
    nota(ax, "equilibrio (s*, t*)", (S, T), (0.7, 0.63))
    etiqueta(ax, 0.86, 0.905, "mejor respuesta de J2:\nt = (1 + s²)/2", color=TINTA, ha="right", va="center")
    etiqueta(ax, 0.515, 0.86, "mejor respuesta\nde J1", color=TINTA, va="center")
    ax.set_xlim(0.5, 1)
    ax.set_ylim(0.5, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("umbral s de J1")
    ax.set_ylabel("umbral t de J2 tras «me planto»")
    formato_coma(ax, 1, 1)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, ticks=[0.25, 0.5, 0.75])
    cb.ax.set_yticklabels(["0,25: favorece\na J2", "0,50: parejo", "0,75: favorece\na J1"], fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(size=0)
    F.guardar(fig, CARPETA, "mapa_silla")


def fig_analogos(R):
    e10 = R["e10"]
    ss = np.linspace(0.5, 0.8, 400)
    so = np.array(e10["oculto"]["curva_s"])
    fig, ax = plt.subplots(figsize=(ANCHO, 3.0))
    ax.axhline(0.5, color=REGLA, lw=0.8)
    ax.plot(ss, A.prob_j1_umbral(ss), color=AZUL)
    ax.plot(so, e10["oculto"]["curva"], color=LADRILLO)
    ax.plot(ss, A.prob_j1_umbral_sin_complemento(ss), color=VERDE)
    for color, txt, dy in ((AZUL, "juego original", 0), (LADRILLO, "A: anuncio secreto", 1), (VERDE, "B: sin complemento", 2)):
        y = 0.4565 - dy * 0.0052
        ax.plot([0.505, 0.52], [y, y], color=color, lw=1.4)
        etiqueta(ax, 0.525, y, txt, color=TINTA2)
    for x0, y0, txt, xt in [
        (S, A.P_GANA_J1, f"{coma(A.P_GANA_J1, 4)} en s* ≈ {coma(S, 3)}", (0.6, 0.462)),
        (A.UMBRAL_OCULTO, 0.5, "½ en 1/√2 ≈ 0,707", (0.715, 0.5035)),
        (A.S_SIN_COMPLEMENTO, A.P_GANA_J1_SIN_COMPLEMENTO,
         f"{coma(A.P_GANA_J1_SIN_COMPLEMENTO, 4)} en {coma(A.S_SIN_COMPLEMENTO, 3)}", (0.585, 0.5035)),
    ]:
        ax.plot([x0], [y0], "o", ms=5.2, color=TINTA, mec=PAPEL, mew=1.2, zorder=5)
        nota(ax, txt, (x0, y0), xt)
    ax.set_xlim(0.495, 0.80)
    ax.set_ylim(0.44, 0.508)
    ax.set_xlabel("umbral de J1")
    ax.set_ylabel("P(J1 gana)")
    formato_coma(ax, 2, 2)
    F.guardar(fig, CARPETA, "analogos")


def main() -> None:
    R = json.loads((RAIZ / "resultados" / "resultados.json").read_text())
    F.aplicar_estilo()
    for fn in (fig_distribucion_m, fig_mc_convergencia, fig_g_de_s, fig_umbrales_j2, fig_decision_j2,
               fig_decision_j1, fig_faroles, fig_telarana, fig_pl_estrategias, fig_pl_convergencia,
               fig_juego_ficticio, fig_mapa_silla, fig_analogos):
        fn(R)
        print("  ok", fn.__name__)
    print(f"Figuras en {CARPETA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
