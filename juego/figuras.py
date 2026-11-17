"""Estilo común de los gráficos (SVG) del entregable.

Los gráficos usan la misma tipografía que el texto (Source Serif 4, corte SmText) y se
guardan con el texto como texto (svg.fonttype = "none"): el navegador lo dibuja con la
fuente incrustada en el HTML, así que figuras y párrafos se ven como una sola pieza.
El fondo es transparente para que se vea el color del papel.

Colores (validados para daltonismo): plantarse = azul, descartar = ladrillo,
complemento = verde agua. Se usan siempre con ese significado.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

AZUL, LADRILLO, VERDE = "#30569a", "#b9532f", "#2a9d8f"
TINTA, TINTA2, GRIS, REGLA, PAPEL = "#1d1c19", "#5e5a51", "#9a958b", "#d6d0c2", "#fdfcf7"
COLOR_ACCION = {0: AZUL, 1: VERDE, 2: LADRILLO}            # P, C, D
NOMBRE_ACCION = {0: "se planta", 1: "complementa", 2: "descarta"}
FUENTE = "Source Serif 4 SmText"
TTF = Path(__file__).resolve().parents[1] / "entregable" / "recursos" / "fuentes" / "ttf"

ANCHO = 6.4        # columna principal (se muestra a 640 px)
ANCHO_DOBLE = 9.2  # columna principal + margen (920 px)


def aplicar_estilo() -> None:
    for ttf in sorted(TTF.glob("*.ttf")):
        font_manager.fontManager.addfont(str(ttf))
    plt.rcParams.update({
        "svg.fonttype": "none",
        "font.family": FUENTE,
        "font.size": 9,
        "mathtext.fontset": "custom",
        "mathtext.rm": FUENTE,
        "mathtext.it": f"{FUENTE}:italic",
        "mathtext.bf": f"{FUENTE}:semibold",
        "mathtext.cal": FUENTE,
        "mathtext.sf": FUENTE,
        "mathtext.tt": FUENTE,
        "mathtext.fallback": "cm",
        "axes.facecolor": "none",
        "figure.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "axes.edgecolor": GRIS,
        "axes.linewidth": 0.6,
        "axes.labelcolor": TINTA2,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.titlecolor": TINTA,
        "axes.titlepad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "xtick.color": GRIS,
        "ytick.color": GRIS,
        "xtick.labelcolor": TINTA2,
        "ytick.labelcolor": TINTA2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.4,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": TINTA2,
        "legend.handlelength": 1.8,
    })


def coma(x: float, d: int = 3) -> str:
    """Número con coma decimal."""
    return f"{x:.{d}f}".replace(".", ",")


def formato_coma(ax, dx: int | None = 2, dy: int | None = 2) -> None:
    if dx is not None:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: coma(v, dx)))
    if dy is not None:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: coma(v, dy)))


def puntos(ax, x, y, color, tam=4.2, **kw):
    """Marcadores chicos con un anillo del color del papel (se leen sobre las líneas)."""
    return ax.plot(x, y, "o", ms=tam, color=color, mec=PAPEL, mew=1.1, zorder=4, **kw)


def nota(ax, texto, xy, xytext, ha="left", va="center", color=TINTA, flecha=True, italica=True, tam=8.8):
    props = dict(arrowstyle="-", color=GRIS, lw=0.6, shrinkA=1, shrinkB=3) if flecha else None
    ax.annotate(texto, xy=xy, xytext=xytext, ha=ha, va=va, fontsize=tam, color=color,
                fontstyle="italic" if italica else "normal", arrowprops=props, zorder=5)


def etiqueta(ax, x, y, texto, color=TINTA2, ha="left", va="center", italica=True, tam=8.8, **kw):
    """Rótulo directo sobre una curva (en lugar de una leyenda)."""
    ax.text(x, y, texto, ha=ha, va=va, color=color, fontsize=tam,
            fontstyle="italic" if italica else "normal", **kw)


def guardar(fig, carpeta: Path, nombre: str) -> Path:
    """Guarda en SVG con identificadores deterministas (diffs limpios en git)."""
    carpeta.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.hashsalt"] = nombre
    ruta = carpeta / f"{nombre}.svg"
    fig.savefig(ruta, format="svg", bbox_inches="tight", pad_inches=0.04, metadata={"Date": None})
    plt.close(fig)
    return ruta
