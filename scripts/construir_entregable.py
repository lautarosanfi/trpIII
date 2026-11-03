#!/usr/bin/env python3
"""Arma entregable/index.html: un único archivo autocontenido (funciona sin internet).

Toma la plantilla entregable/plantilla.html.j2, los números de resultados/resultados.json
y las figuras SVG de resultados/figuras/, e incrusta KaTeX (JS, CSS y fuentes) para que
las fórmulas se vean aunque no haya conexión.

Uso (desde la raíz del repo):
    python scripts/construir_entregable.py
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from jinja2 import Environment, FileSystemLoader, StrictUndefined  # noqa: E402

from juego import analitico as A  # noqa: E402

ENTREGABLE = RAIZ / "entregable"
KATEX = ENTREGABLE / "recursos" / "katex"
FIGURAS = RAIZ / "resultados" / "figuras"


def coma(x, d=3):
    """0.62996 -> '0,630' (texto)."""
    return f"{float(x):.{d}f}".replace(".", ",")


def mcoma(x, d=3):
    """0.62996 -> '0{,}630' (dentro de una fórmula, sin espacio después de la coma)."""
    return f"{float(x):.{d}f}".replace(".", "{,}")


def pct(x, d=1):
    return f"{100 * float(x):.{d}f}".replace(".", ",") + " %"


def miles(n):
    return f"{int(n):,}".replace(",", ".")


def msci(x, d=1):
    """2.82e-05 -> '2{,}8\\times10^{-5}' (notación científica dentro de una fórmula)."""
    mantisa, exponente = f"{float(x):.{d}e}".split("e")
    return f"{mantisa.replace('.', '{,}')}\\times10^{{{int(exponente)}}}"


def katex_css() -> str:
    css = (KATEX / "katex.min.css").read_text()

    def fuente(m):
        nombre = m.group(1)
        datos = base64.b64encode((KATEX / "fonts" / f"{nombre}.woff2").read_bytes()).decode()
        return f'src:url(data:font/woff2;base64,{datos}) format("woff2")'

    # cada @font-face trae woff2, woff y ttf: nos quedamos sólo con woff2, incrustado
    css, n = re.subn(r'src:url\(fonts/([A-Za-z0-9_-]+)\.woff2\) format\("woff2"\)'
                     r'(?:,url\(fonts/[^)]+\) format\("[a-z]+"\))*', fuente, css)
    assert n >= 20, f"sólo se reemplazaron {n} fuentes"
    assert "url(fonts/" not in css
    return css


def svg_en_linea(nombre: str, descripcion: str) -> str:
    svg = (FIGURAS / f"{nombre}.svg").read_text()
    svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg)
    # tamaño adaptable: sin width/height fijos, se conserva el viewBox
    svg = re.sub(r'(<svg[^>]*?)\s(width|height)="[^"]*"', r"\1", svg, count=1)
    svg = re.sub(r'(<svg[^>]*?)\s(width|height)="[^"]*"', r"\1", svg, count=1)
    svg = svg.replace("<svg ", f'<svg role="img" aria-label="{descripcion}" ', 1)
    return svg


FIGS = {
    "mc_convergencia": "Proporción acumulada de victorias de J1 en función del número de partidas",
    "g_de_s": "Probabilidad de que gane J1 en función de su umbral",
    "umbrales_j2": "Umbrales óptimos de J2 en función del umbral de J1",
    "decision_j2": "Decisión de J2 tras un plantarse de J1",
    "decision_j1": "Decisión de J1 contra la estrategia de equilibrio de J2",
    "faroles": "Probabilidad de ganar de J1 con faroles",
    "telarana": "Diagrama de telaraña de las mejores respuestas encadenadas",
    "juego_ficticio": "Convergencia del juego ficticio",
    "pl_estrategias": "Estrategias de equilibrio del juego discretizado",
    "pl_convergencia": "Convergencia del juego discretizado",
    "analogos": "Probabilidad de ganar de J1 en el juego original y en dos variantes",
}


def main() -> None:
    R = json.loads((RAIZ / "resultados" / "resultados.json").read_text())
    env = Environment(loader=FileSystemLoader(ENTREGABLE), undefined=StrictUndefined, autoescape=False,
                      trim_blocks=True, lstrip_blocks=True)
    env.filters.update(c=coma, m=mcoma, pct=pct, miles=miles, msci=msci)
    plantilla = env.get_template("plantilla.html.j2")
    html = plantilla.render(
        R=R,
        A=A,
        S=A.S_ESTRELLA,
        T=A.T_ESTRELLA,
        fig={k: svg_en_linea(k, v) for k, v in FIGS.items()},
        katex_css=katex_css(),
        katex_js=(KATEX / "katex.min.js").read_text(),
        autorender_js=(KATEX / "contrib" / "auto-render.min.js").read_text(),
        fecha=dt.date.today().strftime("%d/%m/%Y"),
    )
    salida = ENTREGABLE / "index.html"
    salida.write_text(html)
    print(f"{salida.relative_to(RAIZ)}  ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
