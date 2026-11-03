"""Estilo común y utilidades para las figuras (SVG) del entregable.

Colores: paleta categórica validada para daltonismo (orden fijo). Las acciones usan
siempre el mismo color en todas las figuras: plantarse = azul, descartar = naranja,
complemento = verde agua.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

AZUL, NARANJA, AGUA, VIOLETA = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
TINTA, TINTA2, APAGADO = "#0b0b0b", "#52514e", "#898781"
GRILLA, EJE, FONDO = "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLOR_ACCION = {0: AZUL, 1: AGUA, 2: NARANJA}          # P, C, D
NOMBRE_ACCION = {0: "plantarse", 1: "complemento", 2: "descartar"}

ANCHO = 6.6  # pulgadas


def aplicar_estilo() -> None:
    plt.rcParams.update({
        "svg.fonttype": "path",
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "mathtext.fontset": "dejavusans",
        "axes.edgecolor": EJE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": TINTA2,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": TINTA,
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRILLA,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": APAGADO,
        "ytick.color": APAGADO,
        "xtick.labelcolor": TINTA2,
        "ytick.labelcolor": TINTA2,
        "figure.facecolor": FONDO,
        "axes.facecolor": FONDO,
        "savefig.facecolor": FONDO,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": TINTA2,
    })


def coma(x: float, d: int = 3) -> str:
    """Número con coma decimal (convención argentina)."""
    return f"{x:.{d}f}".replace(".", ",")


def formato_coma(ax, dx: int = 2, dy: int = 2) -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: coma(v, dx)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: coma(v, dy)))


def puntos(ax, x, y, color, etiqueta=None, tam=6.5, **kw):
    """Marcadores rellenos con anillo del color de fondo (legibles sobre las líneas)."""
    return ax.plot(x, y, "o", ms=tam, color=color, mec=FONDO, mew=1.4, label=etiqueta, zorder=4, **kw)


def nota(ax, texto, xy, xytext, ha="left", va="center", color=TINTA2, flecha=True):
    props = dict(arrowstyle="-", color=APAGADO, lw=0.8, shrinkA=2, shrinkB=3) if flecha else None
    ax.annotate(texto, xy=xy, xytext=xytext, ha=ha, va=va, fontsize=8.5, color=color,
                arrowprops=props, zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", fc=FONDO, ec="none", alpha=0.92))


def guardar(fig, carpeta: Path, nombre: str) -> Path:
    """Guarda en SVG con identificadores deterministas (diffs limpios en git)."""
    carpeta.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.hashsalt"] = nombre
    ruta = carpeta / f"{nombre}.svg"
    fig.savefig(ruta, format="svg", bbox_inches="tight", pad_inches=0.08, metadata={"Date": None})
    plt.close(fig)
    return ruta
