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

from juego import analitico as A  # noqa: E402
from juego import figuras as F  # noqa: E402
from juego.figuras import (AGUA, ANCHO, APAGADO, AZUL, COLOR_ACCION, FONDO, NARANJA,  # noqa: E402
                           TINTA, TINTA2, coma, formato_coma, nota, puntos)

S, T = A.S_ESTRELLA, A.T_ESTRELLA
CARPETA = RAIZ / "resultados" / "figuras"


def fig_mc_convergencia(R):
    tr = R["e9"]["trayectoria"]
    n, p = np.array(tr["n"]), np.array(tr["p"])
    n, p = n[n >= 100], p[n >= 100]
    fig, ax = plt.subplots(figsize=(ANCHO, 2.9))
    nn = np.geomspace(n.min(), n.max(), 300)
    ee = 1.96 * np.sqrt(A.P_GANA_J1 * (1 - A.P_GANA_J1) / nn)
    ax.fill_between(nn, A.P_GANA_J1 - ee, A.P_GANA_J1 + ee, color=AZUL, alpha=0.10, lw=0,
                    label="banda de ±1,96 errores estándar")
    ax.axhline(A.P_GANA_J1, color=TINTA2, lw=1, label=f"valor teórico {coma(A.P_GANA_J1, 4)}")
    ax.plot(n, p, color=AZUL, lw=1.6, label="proporción acumulada de victorias de J1")
    ax.set_xscale("log")
    ax.set_ylim(0.40, 0.58)
    ax.set_xlabel("cantidad de partidas simuladas")
    ax.set_ylabel("P(J1 gana) estimada")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: coma(v, 2)))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", ".")))
    ax.legend(loc="upper right")
    F.guardar(fig, CARPETA, "mc_convergencia")


def fig_g_de_s(R):
    filas = R["e2"]["filas"]
    s = np.array([r["s"] for r in filas])
    p = np.array([r["prob"]["p"] for r in filas])
    ee = np.array([r["prob"]["ee"] for r in filas])
    ss = np.linspace(0.5, 0.8, 400)
    fig, ax = plt.subplots(figsize=(ANCHO, 3.5))
    ax.axhline(0.5, color=APAGADO, lw=0.8)
    ax.text(0.797, 0.5015, "1/2: juego parejo", ha="right", va="bottom", fontsize=8, color=APAGADO)
    ax.plot(ss, A.prob_j1_umbral(ss), color=AZUL, label=r"teoría: $g(s)=\frac{1}{4}+\frac{s}{2}-\frac{s^4}{2}$")
    ax.errorbar(s, p, yerr=1.96 * ee, fmt="none", ecolor=TINTA2, elinewidth=0.8, capsize=0, zorder=3)
    puntos(ax, s, p, AZUL, etiqueta="simulación (J2 aprende de 200.000 partidas)", tam=5.5)
    for x0, txt, dx, dy in [(0.5, "ingenua: nunca descartar\n15/32 ≈ 0,469", 0.018, -0.012),
                            (S, f"óptimo: s* = 4$^{{-1/3}}$ ≈ {coma(S, 3)}\nP = {coma(A.P_GANA_J1, 4)}", 0.03, 0.010),
                            (1 / np.sqrt(2), "1/√2 (umbral óptimo si el anuncio\nfuera secreto) ≈ 0,479", -0.105, -0.026)]:
        y0 = float(A.prob_j1_umbral(x0))
        ax.plot([x0], [y0], "o", ms=7, color=TINTA, mec=FONDO, mew=1.5, zorder=5)
        nota(ax, txt, (x0, y0), (x0 + dx, y0 + dy))
    ax.set_xlim(0.495, 0.80)
    ax.set_ylim(0.44, 0.505)
    ax.set_xlabel("umbral s de J1 (se queda con max(x, 1−x) si es ≥ s; si no, descarta)")
    ax.set_ylabel("P(J1 gana)")
    formato_coma(ax, 2, 3)
    ax.legend(loc="lower left")
    F.guardar(fig, CARPETA, "g_de_s")


def fig_umbrales_j2(R):
    filas = R["e2"]["filas"]
    s = np.array([r["s"] for r in filas])
    up = np.array([(r["umbral_p"] + r["umbral_c"]) / 2 for r in filas])
    ud = np.array([r["umbral_d"] for r in filas])
    ss = np.linspace(0.5, 0.8, 300)
    fig, ax = plt.subplots(figsize=(ANCHO, 3.4))
    ax.plot(ss, (1 + ss) / 2, color=APAGADO, lw=1.4, ls=(0, (4, 3)),
            label="promedio del número de J1 si se plantó: (1+s)/2")
    ax.plot(ss, A.umbral_j2(ss), color=AZUL, label="umbral óptimo de J2 tras «plantarse/complemento»: (1+s²)/2")
    puntos(ax, s, up, AZUL, tam=5.5)
    ax.plot(ss, np.full_like(ss, 0.5), color=NARANJA, label="umbral óptimo de J2 tras «descartar»: 1/2")
    puntos(ax, s, ud, NARANJA, tam=5.5)
    ax.plot([], [], "o", ms=5.5, color=TINTA2, mec=FONDO, label="aprendidos por J2 en la simulación")
    ax.axvline(S, color=APAGADO, lw=0.8)
    nota(ax, f"con s = s*: J2 se planta desde\nt* = {coma(T, 3)}, aunque el promedio\nde J1 sea {coma((1 + S) / 2, 3)}",
         (S, T), (S + 0.035, 0.58))
    ax.set_xlim(0.495, 0.80)
    ax.set_ylim(0.47, 0.93)
    ax.set_xlabel("umbral s de J1")
    ax.set_ylabel("umbral de J2 (sobre max(z, 1−z))")
    formato_coma(ax, 2, 2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)
    F.guardar(fig, CARPETA, "umbrales_j2")


def fig_decision_j2(R):
    m = np.linspace(0.5, 1, 400)
    gana_pl = np.clip((m - S) / (1 - S), 0, 1)
    gana_d = (1 - S) / 2
    fig, ax = plt.subplots(figsize=(ANCHO, 3.3))
    for x0, txt in [(0.625, "0,625"), (0.75, "0,75"), ((1 + S) / 2, "promedio\n0,815")]:
        ax.axvline(x0, color=F.EJE, lw=0.9, zorder=1)
        ax.text(x0, 1.02, txt, ha="center", va="bottom", fontsize=8, color=TINTA2)
    ax.plot(m, gana_pl, color=AZUL, label="plantarse con m: P(V1 < m | J1 se plantó) = (m − s*)/(1 − s*)")
    ax.plot(m, np.full_like(m, gana_d), color=NARANJA, label="descartar: 1 − E[V1 | J1 se plantó] = (1 − s*)/2")
    ax.plot([T], [gana_d], "o", ms=7.5, color=TINTA, mec=FONDO, mew=1.5, zorder=5)
    nota(ax, f"cruce en t* = (1+s*²)/2 ≈ {coma(T, 3)}\n(ganar: {coma(gana_d, 3)} en ambos casos)",
         (T, gana_d), (0.995, 0.075), ha="right")
    ax.set_xlim(0.5, 1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("m = max(z, 1−z): lo mejor que J2 puede quedarse sin descartar")
    ax.set_ylabel("P(J2 gana)")
    formato_coma(ax, 2, 1)
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0, 0.93), frameon=True, facecolor=FONDO,
              edgecolor="none", framealpha=1)
    F.guardar(fig, CARPETA, "decision_j2")


def fig_decision_j1(R):
    m = np.linspace(0.5, 1, 400)
    fig, ax = plt.subplots(figsize=(ANCHO, 3.3))
    ax.plot(m, A.pago_plantarse(m, T), color=AZUL,
            label="plantarse con m: s*²·m + 2(m − t*)⁺  (J2 descarta con prob. s*² ≈ 0,397)")
    ax.plot(m, np.full_like(m, 0.25), color=NARANJA, label="descartar: 1/4 (tras un descarte J2 nunca descarta)")
    ax.plot([S], [0.25], "o", ms=7.5, color=TINTA, mec=FONDO, mew=1.5, zorder=5)
    nota(ax, f"indiferencia: s*·s*² = 1/4  ⇒  s* = 4$^{{-1/3}}$ ≈ {coma(S, 3)}", (S, 0.25), (S + 0.06, 0.12))
    for x0 in [0.52, 0.55]:
        y0 = float(A.pago_plantarse(x0, T))
        ax.plot([x0], [y0], "o", ms=6, color=AZUL, mec=FONDO, mew=1.4, zorder=5)
    nota(ax, f"«farol» con 0,52: {coma(float(A.pago_plantarse(0.52, T)), 3)} < 0,25\ncon 0,55: "
             f"{coma(float(A.pago_plantarse(0.55, T)), 3)} < 0,25", (0.52, float(A.pago_plantarse(0.52, T))),
         (0.54, 0.45))
    nota(ax, "desde t* también le gana a\nun J2 que se plantó", (T, float(A.pago_plantarse(T, T))), (0.79, 0.37))
    ax.set_xlim(0.5, 1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("m = max(x, 1−x)")
    ax.set_ylabel("P(J1 gana)")
    formato_coma(ax, 2, 1)
    ax.legend(loc="upper left", fontsize=8)
    F.guardar(fig, CARPETA, "decision_j1")


def fig_faroles(R):
    e5 = R["e5"]
    fig, axs = plt.subplots(1, 2, figsize=(ANCHO, 3.2), sharey=True)
    for ax, clave, xlab, tit in [
        (axs[0], "aleatorio", "q: prob. de plantarse igual si max(x,1−x) < s*", "a) farol al azar"),
        (axs[1], "polarizado", "b: también se planta si max(x,1−x) < b", "b) farol con los peores números"),
    ]:
        filas = e5[clave]
        xv = np.array([r["q" if clave == "aleatorio" else "b"] for r in filas])
        ax.axhline(A.P_GANA_J1, color=APAGADO, lw=0.8)
        ax.plot(xv, [r["vs_t_estrella"] for r in filas], color=APAGADO, lw=1.8,
                label="J2 no lo anticipa (sigue con t*)")
        ax.plot(xv, [r["vs_mejor_respuesta"] for r in filas], color=TINTA, lw=2,
                label="J2 se adapta (mejor respuesta)")
        ax.set_title(tit)
        ax.set_xlabel(xlab, fontsize=8.5)
        formato_coma(ax, 1 if clave == "aleatorio" else 2, 3)
    mc = e5["mc_farol_03"]["mc"]
    axs[0].errorbar([0.3], [mc["p"]], yerr=[1.96 * mc["ee"]], fmt="none", ecolor=TINTA2, elinewidth=0.8)
    puntos(axs[0], [0.3], [mc["p"]], AZUL, etiqueta="simulación (J2 aprende)", tam=6)
    axs[0].text(0.99, A.P_GANA_J1 + 0.0006, "equilibrio (sin farol)", ha="right", va="bottom", fontsize=8,
                color=APAGADO)
    axs[0].set_ylabel("P(J1 gana)")
    axs[0].set_ylim(0.466, 0.4885)
    axs[0].yaxis.set_major_locator(plt.MultipleLocator(0.005))
    axs[1].set_xlim(0.5, S)
    axs[1].xaxis.set_major_locator(plt.MultipleLocator(0.05))
    fig.subplots_adjust(wspace=0.12)
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.2), ncol=3, fontsize=8)
    F.guardar(fig, CARPETA, "faroles")


def fig_telarana(R):
    s_seq = np.array(R["e6"]["s"])
    fig, axs = plt.subplots(1, 2, figsize=(ANCHO, 3.2), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]
    ss = np.linspace(0.5, 0.75, 400)
    phi = np.array([A.mejor_respuesta_j1(A.umbral_j2(x)) for x in ss])
    ax.plot(ss, ss, color=APAGADO, lw=1, label="diagonal")
    ax.plot(ss, phi, color=AZUL, label="φ(s): mejor resp. de J1 a la de J2")
    xs, ys = [s_seq[0]], [s_seq[0]]
    for k in range(10):
        xs += [s_seq[k], s_seq[k + 1]]
        ys += [s_seq[k + 1], s_seq[k + 1]]
    ax.plot(xs, ys, color=NARANJA, lw=1.2, label="iteración desde s = 1/2")
    ax.plot([S], [S], "o", ms=7, color=TINTA, mec=FONDO, mew=1.5, zorder=5)
    nota(ax, "s*: punto fijo\n(φ'(s*) = −2)", (S, S), (0.515, 0.715))
    ax.set_xlim(0.5, 0.75)
    ax.set_ylim(0.5, 0.75)
    ax.set_aspect("equal")
    ax.set_title("a) diagrama de telaraña")
    ax.set_xlabel(r"$s_k$")
    ax.set_ylabel(r"$s_{k+1}$")
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    formato_coma(ax, 2, 2)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.05, -0.2), fontsize=7.5)
    ax = axs[1]
    k = np.arange(len(s_seq))
    ax.axhline(S, color=APAGADO, lw=0.8)
    ax.text(len(k) - 1, S + 0.004, "s*", ha="right", va="bottom", fontsize=8.5, color=TINTA2)
    ax.plot(k, s_seq, color=NARANJA, lw=1.2, marker="o", ms=4.5, mec=FONDO, mew=1)
    ax.set_title("b) el umbral de J1 nunca se asienta")
    ax.set_xlabel("ronda k")
    ax.set_ylabel("umbral de J1")
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: coma(v, 2)))
    fig.subplots_adjust(wspace=0.35)
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
                # un leve solapamiento con el tramo siguiente evita «costuras» blancas de antialiasing
                ax.add_patch(plt.Rectangle((x0, abajo), w + 0.6 * ancho_celda, alto, color=COLOR_ACCION[a], lw=0))
            abajo += alto
    for u in umbrales:
        ax.axvline(u, color=TINTA, lw=0.9, ls=(0, (3, 2)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.grid(False)
    for lado in ("left", "top", "right"):
        ax.spines[lado].set_visible(False)
    if etiquetas:
        accion = probs.argmax(axis=1)
        cortes = np.flatnonzero(np.diff(accion)) + 1
        for seg in np.split(np.arange(len(x)), cortes):
            a = accion[seg[0]]
            ancho = x[seg[-1]] - x[seg[0]]
            if ancho > 0.12:
                ax.text(x[seg].mean(), 0.5, F.NOMBRE_ACCION[a], ha="center", va="center", fontsize=8.5,
                        color="white", fontweight="bold")


def fig_pl_estrategias(R):
    e7 = R["e7"]
    n = e7["n"]
    x = (np.arange(n) + 0.5) / n
    r = np.array(e7["estrategia_j1"])
    q = np.array(e7["estrategia_j2"])
    fig, axs = plt.subplots(4, 1, figsize=(ANCHO, 3.9), sharex=True, gridspec_kw={"hspace": 0.75})
    filas = [(r, (1 - S, S), "J1, según su número x"),
             (q[0], (1 - T, T), "J2 si J1 anunció «plantarse», según su número z"),
             (q[1], (1 - T, T), "J2 si J1 anunció «complemento»"),
             (q[2], (0.5,), "J2 si J1 anunció «descartar»")]
    for ax, (pr, umb, tit) in zip(axs, filas):
        _franja(ax, x, pr, umb)
        ax.set_title(tit, fontsize=8.8, fontweight="normal", color=TINTA2, pad=3)
    axs[-1].set_xticks([0, 1 - T, 1 - S, 0.5, S, T, 1])
    axs[-1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: coma(v, 3) if 0 < v < 1 else coma(v, 0)))
    axs[-1].tick_params(axis="x", labelsize=7.5)
    for lab in axs[-1].get_xticklabels():
        lab.set_rotation(0)
    F.guardar(fig, CARPETA, "pl_estrategias")


def fig_pl_convergencia(R):
    conv = R["e7"]["convergencia"]
    n = np.array([c["n"] for c in conv])
    fig, axs = plt.subplots(1, 2, figsize=(ANCHO, 2.9))
    ax = axs[0]
    ax.axhline(A.P_GANA_J1, color=APAGADO, lw=0.8)
    ax.text(n[-1], A.P_GANA_J1 - 0.000008, f"juego continuo: {coma(A.P_GANA_J1, 5)}", ha="right", va="top",
            fontsize=8, color=TINTA2)
    ax.plot(n, [c["valor"] for c in conv], color=AZUL, marker="o", ms=5.5, mec=FONDO, mew=1.2)
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([str(v) for v in n])
    ax.minorticks_off()
    ax.set_title("a) valor del juego discreto")
    ax.set_xlabel("N (puntos de la grilla)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: coma(v, 5)))
    ax = axs[1]
    ax.axhline(S, color=APAGADO, lw=0.8)
    ax.axhline(T, color=APAGADO, lw=0.8)
    ax.plot(n, [c["descarte_j1"][1] for c in conv], color=AZUL, marker="o", ms=5.5, mec=FONDO, mew=1.2,
            label="J1: mayor x que descarta")
    ax.plot(n, [c["descarte_j2_tras_p"][1] for c in conv], color=NARANJA, marker="o", ms=5.5, mec=FONDO,
            mew=1.2, label="J2 tras «plantarse»: mayor z que descarta")
    ax.text(n[-1], S + 0.004, f"s* ≈ {coma(S, 3)}", va="bottom", ha="right", fontsize=8, color=TINTA2)
    ax.text(n[-1], T + 0.004, f"t* ≈ {coma(T, 3)}", va="bottom", ha="right", fontsize=8, color=TINTA2)
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([str(v) for v in n])
    ax.minorticks_off()
    ax.set_ylim(0.53, 0.73)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.set_title("b) umbrales")
    ax.set_xlabel("N (puntos de la grilla)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: coma(v, 2)))
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.22), fontsize=7.5)
    fig.subplots_adjust(wspace=0.3)
    F.guardar(fig, CARPETA, "pl_convergencia")


def fig_juego_ficticio(R):
    e8 = R["e8"]
    fig, axs = plt.subplots(1, 2, figsize=(ANCHO, 3.0), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axs[0]
    k, b = np.array(e8["iteracion"]), np.array(e8["brecha"])
    ax.plot(k, b, color=AZUL)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("a) brecha de explotabilidad")
    ax.set_xlabel("ronda")
    ax.set_ylabel("mejor resp. a J2 − mejor resp. a J1")
    ax = axs[1]
    r = np.array(e8["estrategia_j1"])
    x = (np.arange(len(r)) + 0.5) / len(r)
    _franja(ax, x, r, (1 - S, S), etiquetas=False)
    ax.set_yticks([0, 0.5, 1])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: coma(v, 1)))
    ax.set_title(f"b) estrategia promedio de J1 tras {k[-1]:,} rondas".replace(",", "."))
    ax.set_xlabel("x (número de J1)")
    ax.set_ylabel("probabilidad")
    for a, nombre in ((0, "plantarse"), (1, "complemento"), (2, "descartar")):
        ax.fill_between([], [], color=COLOR_ACCION[a], label=nombre)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: coma(v, 1)))
    F.guardar(fig, CARPETA, "juego_ficticio")


def fig_analogos(R):
    e10 = R["e10"]
    ss = np.linspace(0.5, 0.8, 400)
    so = np.array(e10["oculto"]["curva_s"])
    fig, ax = plt.subplots(figsize=(ANCHO, 3.3))
    ax.axhline(0.5, color=APAGADO, lw=0.8)
    ax.plot(ss, A.prob_j1_umbral(ss), color=AZUL, label="original (J2 ve el anuncio)")
    ax.plot(so, e10["oculto"]["curva"], color=NARANJA, label="A) anuncio secreto")
    ax.plot(ss, A.prob_j1_umbral_sin_complemento(ss), color=AGUA, label="B) sin complemento")
    for x0, y0, txt, xt in [
        (S, A.P_GANA_J1, f"{coma(A.P_GANA_J1, 4)} en s* ≈ {coma(S, 3)}", (0.60, 0.464)),
        (A.UMBRAL_OCULTO, 0.5, "1/2 en s = 1/√2 ≈ 0,707", (0.728, 0.5035)),
        (A.S_SIN_COMPLEMENTO, A.P_GANA_J1_SIN_COMPLEMENTO,
         f"{coma(A.P_GANA_J1_SIN_COMPLEMENTO, 4)} en s ≈ {coma(A.S_SIN_COMPLEMENTO, 3)}", (0.585, 0.503)),
    ]:
        ax.plot([x0], [y0], "o", ms=7, color=TINTA, mec=FONDO, mew=1.5, zorder=5)
        nota(ax, txt, (x0, y0), xt)
    ax.set_xlim(0.495, 0.80)
    ax.set_ylim(0.44, 0.508)
    ax.set_xlabel("umbral de J1")
    ax.set_ylabel("P(J1 gana)")
    formato_coma(ax, 2, 3)
    ax.legend(loc="lower left")
    F.guardar(fig, CARPETA, "analogos")


def main() -> None:
    R = json.loads((RAIZ / "resultados" / "resultados.json").read_text())
    F.aplicar_estilo()
    for fn in (fig_mc_convergencia, fig_g_de_s, fig_umbrales_j2, fig_decision_j2, fig_decision_j1,
               fig_faroles, fig_telarana, fig_pl_estrategias, fig_pl_convergencia, fig_juego_ficticio,
               fig_analogos):
        fn(R)
        print("  ok", fn.__name__)
    print(f"Figuras en {CARPETA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
