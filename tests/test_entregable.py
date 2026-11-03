"""Chequeos del entregable HTML (sin navegador)."""

import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PLANTILLA = RAIZ / "entregable" / "plantilla.html.j2"
MATH = re.compile(r"\\\((.*?)\\\)|\\\[(.*?)\\\]", re.S)


def _cuerpo() -> str:
    texto = PLANTILLA.read_text()
    return texto[: texto.index("<script>{{ katex_js }}</script>")]


def test_no_hay_menor_o_mayor_crudos_dentro_de_formulas():
    """Un '<' seguido de letra dentro de \\( … \\) el navegador lo toma como etiqueta HTML y se
    «come» el texto que sigue. Dentro de las fórmulas hay que escribir &lt; y &gt;."""
    malos = [m.group(0)[:60] for m in MATH.finditer(_cuerpo()) if "<" in m.group(0) or ">" in m.group(0)]
    assert malos == []


def test_delimitadores_balanceados():
    cuerpo = _cuerpo()
    assert cuerpo.count("\\(") == cuerpo.count("\\)")
    assert cuerpo.count("\\[") == cuerpo.count("\\]")
