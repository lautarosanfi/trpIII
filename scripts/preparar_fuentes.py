#!/usr/bin/env python3
"""Recorta la tipografía Source Serif 4 para incrustarla en el entregable.

No hace falta correrlo: los archivos resultantes ya están en entregable/recursos/fuentes/.
Queda para documentar de dónde salen y poder regenerarlos.

Origen: paquete npm «source-serif» 4.5.1 (Adobe, licencia SIL Open Font License 1.1).
    npm pack source-serif@4.5.1 && tar xzf source-serif-4.5.1.tgz
    python scripts/preparar_fuentes.py package/TTF

Se conservan las funciones OpenType que usa el documento (versalitas, números de estilo
antiguo y alineados, fracciones, kerning, ligaduras) y sólo los caracteres latinos, la
puntuación española y algunos símbolos matemáticos. Requiere fonttools y brotli.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fontTools import subset

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "entregable" / "recursos" / "fuentes"

# (archivo de origen, nombre de familia CSS, peso, estilo)
CORTES = [
    ("SourceSerif4-Regular", "Source Serif 4", 400, "normal"),
    ("SourceSerif4-It", "Source Serif 4", 400, "italic"),
    ("SourceSerif4-Semibold", "Source Serif 4", 600, "normal"),
    ("SourceSerif4-SemiboldIt", "Source Serif 4", 600, "italic"),
    ("SourceSerif4Display-Regular", "Source Serif 4 Display", 400, "normal"),
    ("SourceSerif4Display-It", "Source Serif 4 Display", 400, "italic"),
    ("SourceSerif4SmText-Regular", "Source Serif 4 SmText", 400, "normal"),
    ("SourceSerif4SmText-It", "Source Serif 4 SmText", 400, "italic"),
    ("SourceSerif4SmText-Semibold", "Source Serif 4 SmText", 600, "normal"),
]
# los gráficos de matplotlib necesitan TTF (para medir el texto); usan el corte SmText
PARA_GRAFICOS = {"SourceSerif4SmText-Regular", "SourceSerif4SmText-It", "SourceSerif4SmText-Semibold"}

UNICODES = ("U+0020-007E,U+00A0-00FF,U+0131,U+0152-0153,U+02C6,U+02DA,U+02DC,U+0370-03FF,U+2009-200A,"
            "U+2010-2015,U+2018-201E,U+2020-2022,U+2026,U+202F,U+2030,U+2032-2033,U+2039-203A,"
            "U+2044,U+2070-209F,U+20AC,U+2122,U+2153-215E,U+2190-2199,U+2200-22FF,U+25A1,U+25CB")
FUNCIONES = ("kern,liga,clig,calt,ccmp,locl,mark,mkmk,smcp,c2sc,onum,lnum,pnum,tnum,"
             "frac,numr,dnom,sups,subs,sinf,case,zero,ordn")


def recortar(origen: Path, salida: Path, formato: str | None) -> None:
    args = [str(origen), f"--output-file={salida}", f"--unicodes={UNICODES}",
            f"--layout-features={FUNCIONES}", "--no-hinting", "--desubroutinize",
            "--name-IDs=*", "--name-legacy", "--name-languages=*"]
    if formato:
        args.append(f"--flavor={formato}")
    subset.main(args)


def main() -> None:
    carpeta = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("package/TTF")
    (DESTINO / "ttf").mkdir(parents=True, exist_ok=True)
    for nombre, *_ in CORTES:
        origen = carpeta / f"{nombre}.ttf"
        recortar(origen, DESTINO / f"{nombre}.woff2", "woff2")
        if nombre in PARA_GRAFICOS:
            recortar(origen, DESTINO / "ttf" / f"{nombre}.ttf", None)
        print("ok", nombre)


if __name__ == "__main__":
    main()
