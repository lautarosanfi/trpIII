"""Diagramas dibujados a mano en SVG (estilo TikZ) para el entregable.

Se insertan en el HTML tal cual: el texto usa la tipografía del documento y los colores
salen de las variables CSS de la página (clases .diag, .caja, .flecha, etc.).
Cada diagrama define su propia punta de flecha con un id único.
"""

from __future__ import annotations

from . import analitico as A

S, T = A.S_ESTRELLA, A.T_ESTRELLA
AZUL, LADRILLO, VERDE = "#30569a", "#b9532f", "#2a9d8f"


def _c(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def _punta(ident: str) -> str:
    return (f'<defs><marker id="{ident}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" '
            f'markerHeight="6.5" orient="auto-start-reverse"><path d="M0,1 L10,5 L0,9 z" class="punta"/>'
            f"</marker></defs>")


def _caja(x, y, w, h, lineas, clase="caja", color=None):
    borde = f' style="stroke:{color};stroke-width:1.3"' if color else ""
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" class="{clase}"{borde}/>']
    n = len(lineas)
    y0 = y + h / 2 - (n - 1) * 8.5 + 4.5
    for i, ln in enumerate(lineas):
        out.append(f'<text x="{x + w / 2}" y="{y0 + i * 17:.1f}" text-anchor="middle">{ln}</text>')
    return "\n".join(out)


def partida() -> str:
    """Cómo transcurre una partida."""
    p = "pa"
    o = [f'<svg viewBox="0 0 700 214" class="diag" role="img" aria-label="Cómo transcurre una partida">', _punta(p)]
    o.append('<text x="16" y="67" class="rotulo">J1</text><text x="16" y="177" class="rotulo">J2</text>')
    o.append(_caja(50, 36, 118, 54, ["recibe", '<tspan class="it">x</tspan> ~ U(0,1)']))
    o.append(_caja(182, 36, 118, 54, ["elige", "P, C o D"]))
    o.append(_caja(314, 36, 118, 54, ["anuncia", "la acción"]))
    o.append(_caja(314, 146, 118, 54, ['recibe <tspan class="it">z</tspan>', "y oye el anuncio"]))
    o.append(_caja(446, 146, 118, 54, ["elige", "P, C o D"]))
    o.append(_caja(578, 36, 118, 164, ["se muestran", "los números:", "gana el mayor"]))
    for x1, x2, y in ((168, 182, 63), (300, 314, 63), (432, 446, 173), (564, 578, 173), (432, 578, 63)):
        o.append(f'<path d="M{x1},{y} L{x2 - 1},{y}" class="flecha" marker-end="url(#{p})"/>')
    o.append(f'<path d="M373,90 L373,145" class="flecha discontinua" marker-end="url(#{p})"/>')
    o.append('<text x="363" y="113" text-anchor="end" class="chico it">se anuncia la acción,</text>'
             '<text x="363" y="129" text-anchor="end" class="chico it">no el número</text>')
    o.append('<text x="505" y="56" text-anchor="middle" class="chico it">número final de J1</text>')
    o.append("</svg>")
    return "\n".join(o)


def arbol() -> str:
    """Árbol de probabilidades de una partida con ambos jugadores en equilibrio."""
    r = A.resumen()
    p1, d1 = r["p_j1_planta"], r["p_j1_descarta"]
    q = r["p_j2_descarta_si_j1_planto"]
    g11, g12 = r["p_gana_j1_ambos_plantados"], r["p_gana_j1_planto_y_j2_descarto"]
    o = ['<svg viewBox="0 0 700 236" class="diag" role="img" aria-label="Árbol de probabilidades de una partida">']
    nodos = {"R": (40, 128), "N1": (250, 76), "N2": (250, 192), "N11": (462, 40), "N12": (462, 112),
             "N21": (462, 192)}
    for a, b, txt, dy in (("R", "N1", _c(p1), -9), ("R", "N2", _c(d1), 18), ("N1", "N11", _c(1 - q), -9),
                          ("N1", "N12", _c(q), 18), ("N2", "N21", "1", -9)):
        (x1, y1), (x2, y2) = nodos[a], nodos[b]
        o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="rama"/>')
        o.append(f'<text x="{(x1 + x2) / 2:.0f}" y="{(y1 + y2) / 2 + dy:.0f}" text-anchor="middle" class="it prob">{txt}</text>')
    for x, y in nodos.values():
        o.append(f'<circle cx="{x}" cy="{y}" r="3.2" class="nodo"/>')
    o.append('<text x="40" y="112" text-anchor="middle" class="chico it">partida</text>')
    o.append('<text x="238" y="62" text-anchor="end">J1 se planta o complementa</text>')
    o.append('<text x="250" y="214" text-anchor="middle">J1 descarta</text>')
    o.append('<text x="462" y="25" text-anchor="middle">J2 se planta</text>')
    o.append('<text x="462" y="134" text-anchor="middle">J2 descarta</text>')
    o.append('<text x="462" y="214" text-anchor="middle">J2 se planta</text>')
    for y, g in ((40, g11), (112, g12), (192, 0.25)):
        o.append(f'<text x="484" y="{y + 4.5}" class="hoja">J1 gana con prob. <tspan class="num">{_c(g)}</tspan></text>')
    o.append("</svg>")
    return "\n".join(o)


def rectas() -> str:
    """Qué hace cada jugador según dónde cae su número (rectas numéricas)."""
    X = lambda v: 40 + 620 * v  # noqa: E731
    filas = [
        ("Jugador 1, según su número x", [(0, 1 - S, VERDE, "complementa"), (1 - S, S, LADRILLO, "descarta"),
                                          (S, 1, AZUL, "se planta")], [0, 1 - S, S, 1]),
        ("Jugador 2 si J1 anunció «me planto» o «complemento», según su número z",
         [(0, 1 - T, VERDE, "complementa"), (1 - T, T, LADRILLO, "descarta"), (T, 1, AZUL, "se planta")],
         [0, 1 - T, T, 1]),
        ("Jugador 2 si J1 anunció «descarto», según su número z",
         [(0, 0.5, VERDE, "complementa"), (0.5, 1, AZUL, "se planta")], [0, 0.5, 1]),
    ]
    o = ['<svg viewBox="0 0 700 296" class="diag" role="img" aria-label="Estrategias óptimas como rectas numéricas">']
    for i, (titulo, tramos, marcas) in enumerate(filas):
        y = 18 + i * 100
        o.append(f'<text x="40" y="{y}" class="it">{titulo}</text>')
        for a, b, color, nombre in tramos:
            x1, x2 = X(a) + (1.5 if a > 0 else 0), X(b) - (1.5 if b < 1 else 0)
            o.append(f'<line x1="{x1:.1f}" y1="{y + 40}" x2="{x2:.1f}" y2="{y + 40}" stroke="{color}" stroke-width="7"/>')
            o.append(f'<text x="{(x1 + x2) / 2:.1f}" y="{y + 29}" text-anchor="middle" class="chico it">{nombre}</text>')
        for v in marcas:
            o.append(f'<line x1="{X(v):.1f}" y1="{y + 45}" x2="{X(v):.1f}" y2="{y + 50}" class="rama"/>')
            etiqueta = "0" if v == 0 else "1" if v == 1 else _c(v, 3 if v != 0.5 else 1)
            o.append(f'<text x="{X(v):.1f}" y="{y + 64}" text-anchor="middle" class="chico num">{etiqueta}</text>')
    o.append("</svg>")
    return "\n".join(o)


def flujo() -> str:
    """Diagrama de flujo de las estrategias óptimas."""
    p = "fl"
    o = [f'<svg viewBox="0 0 700 626" class="diag" role="img" aria-label="Diagrama de flujo de las estrategias óptimas">',
         _punta(p)]
    fl = lambda d: f'<path d="{d}" class="flecha" marker-end="url(#{p})"/>'  # noqa: E731

    def decision(cx, cy, texto):
        pts = f"{cx - 90},{cy - 23} {cx + 90},{cy - 23} {cx + 110},{cy} {cx + 90},{cy + 23} {cx - 90},{cy + 23} {cx - 110},{cy}"
        return f'<polygon points="{pts}" class="caja"/><text x="{cx}" y="{cy + 5}" text-anchor="middle" class="it">{texto}</text>'

    o.append('<text x="10" y="22" class="rotulo">Jugador 1</text>')
    o.append(_caja(230, 34, 240, 38, ['recibe <tspan class="it">x</tspan> ~ U(0,1)']))
    o.append(fl("M350,72 L350,95"))
    o.append(decision(350, 121, "¿dónde cae x?"))
    o.append(fl("M262,134 C190,150 125,160 115,196"))
    o.append('<text x="62" y="170" class="chico it">x ≤ 1 − s*</text>')
    o.append(fl("M350,144 L350,196"))
    o.append('<text x="358" y="176" class="chico it">1 − s* &lt; x &lt; s*</text>')
    o.append(fl("M438,134 C510,150 575,160 585,196"))
    o.append('<text x="600" y="176" class="chico it">x ≥ s*</text>')
    o.append(_caja(20, 198, 190, 70, ["complementa", "queda con 1 − x", "anuncia «complemento»"], color=VERDE))
    o.append(_caja(255, 198, 190, 70, ["descarta", "queda con un y nuevo", "anuncia «descarto»"], color=LADRILLO))
    o.append(_caja(490, 198, 190, 70, ["se planta", "queda con x", "anuncia «me planto»"], color=AZUL))
    o.append('<line x1="10" y1="302" x2="690" y2="302" class="rama discontinua"/>')
    o.append('<text x="350" y="294" text-anchor="middle" class="chico it">J2 oye el anuncio, no el número</text>')
    o.append('<text x="10" y="330" class="rotulo">Jugador 2</text>')
    o.append(_caja(205, 342, 290, 38, ['recibe <tspan class="it">z</tspan> ~ U(0,1) y oye el anuncio']))
    o.append(fl("M350,380 L350,403"))
    o.append(decision(350, 429, "¿qué anunció J1?"))
    o.append(fl("M262,442 C228,460 200,468 190,482"))
    o.append('<text x="150" y="462" text-anchor="end" class="chico it">«descarto»</text>')
    o.append(fl("M438,442 C472,460 500,468 510,482"))
    o.append('<text x="528" y="455" class="chico it">«me planto» o</text>'
             '<text x="528" y="471" class="chico it">«complemento»</text>')
    o.append(_caja(95, 484, 190, 34, ["u = ½"]))
    o.append(_caja(415, 484, 190, 34, [f"u = t* ≈ {_c(T)}"]))
    o.append(fl("M190,518 C200,532 248,536 266,540"))
    o.append(fl("M510,518 C500,532 452,536 434,540"))
    o.append(decision(350, 565, "¿dónde cae z?"))
    o.append(f'<text x="128" y="570" text-anchor="middle" class="chico"><tspan class="it">z ≤ 1 − u</tspan>: complementa</text>')
    o.append(f'<text x="575" y="570" text-anchor="middle" class="chico"><tspan class="it">z ≥ u</tspan>: se planta</text>')
    o.append('<text x="350" y="606" text-anchor="middle" class="chico"><tspan class="it">1 − u &lt; z &lt; u</tspan>: '
             'descarta (si u = ½, nunca ocurre)</text>')
    o.append("</svg>")
    return "\n".join(o)


DIAGRAMAS = {"partida": partida, "arbol": arbol, "rectas": rectas, "flujo": flujo}
