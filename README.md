# El Juego de los Números Uniformes — TRP III (2026)

Trabajo final del Taller de Resolución de Problemas III (Universidad Austral). Enunciado y consignas: [`consigna/final_trp3_2026.pdf`](consigna/final_trp3_2026.pdf).

**El entregable es [`entregable/index.html`](entregable/index.html)**: un único archivo autocontenido (fórmulas, figuras y fuentes incrustadas; se abre con cualquier navegador, sin internet).

## El problema en una línea

Dos jugadores reciben cada uno un número U(0,1) y, en orden, eligen **plantarse**, **complementar** (x → 1 − x) o **descartar** (número nuevo). J1 anuncia su acción (no su número) antes de que juegue J2. Gana el número más alto.

## Respuesta

| | |
|---|---|
| Estrategia de J1 | quedarse con max(x, 1 − x) si es ≥ s\* = 4^(−1/3) ≈ 0,630; si no, descartar |
| Estrategia de J2 | tras «descarto»: quedarse siempre con max(z, 1 − z). Tras «me planto» o «complemento»: quedarse si max(z, 1 − z) ≥ t\* = (1 + s\*²)/2 ≈ 0,698; si no, descartar |
| Ventaja | **J2**: gana con probabilidad 3/4 − 3/(8·∛4) ≈ **0,5138** (J1: ≈ 0,4862) |

Es el único equilibrio (salvo casos de probabilidad cero) y ningún farol lo mejora. Se obtuvo por tres caminos independientes: simulación (J2 *aprende* su respuesta a partir de partidas), demostración analítica y programación lineal sobre el juego discretizado sin suponer la forma de las estrategias.

## Estructura

```
consigna/                   enunciado de la cátedra
juego/                      paquete Python
  modelo.py                 reglas, estrategias (vectores de probabilidad) y simulador Monte Carlo
  analitico.py              fórmulas cerradas: equilibrio, mejores respuestas, variantes
  numerico.py               mejor respuesta de J2 a cualquier estrategia de J1 (exacta y aprendida de datos)
  discreto.py               juego discretizado: programación lineal (equilibrio exacto) y juego ficticio
  figuras.py                estilo común de los gráficos
scripts/
  explorar.py               experimentos E1–E10 → resultados/resultados.json
  graficar.py               figuras → resultados/figuras/*.svg
  construir_entregable.py   arma entregable/index.html a partir de la plantilla
resultados/                 salidas generadas (se versionan para poder leerlas sin correr nada)
entregable/
  plantilla.html.j2         texto del entregable (HTML + Jinja2 + fórmulas KaTeX)
  index.html                ENTREGABLE (generado)
  recursos/katex/           KaTeX 0.16.11 (licencia MIT), para ver las fórmulas sin conexión
tests/                      verificaciones automáticas (pytest)
```

## Cómo reproducir

Requiere Python 3.11 o superior.

```bash
pip install -r requirements.txt
make todo          # = explorar + graficar + entregable + test
```

o paso a paso:

```bash
python scripts/explorar.py              # ~30 s; imprime un resumen de cada experimento
python scripts/graficar.py
python scripts/construir_entregable.py
python -m pytest
```

`python scripts/explorar.py --rapido` corre con 10 veces menos partidas (sirve para probar cambios; sobrescribe `resultados/resultados.json`, así que antes de armar el entregable conviene volver a correr la versión completa). Todas las simulaciones usan semillas fijas.

## Experimentos de la exploración

| | Pregunta | Resultado |
|---|---|---|
| E1 | ¿Sirve «complementar si x < 0,5, nunca descartar»? | J1 gana 15/32 ≈ 0,469: J2 lo explota con umbral 5/8 |
| E2 | ¿Conviene descartar números medianos? | Sí: el óptimo es descartar si max(x, 1−x) < 0,630 |
| E3 | ¿A qué apunta J2 tras «me planto»? | A un cuantil, no al promedio; «superar el promedio» le hace perder |
| E4–E5 | ¿Sirve el farol? | No: ni puntual (0,206 < 0,25) ni sistemático, y si J2 lo detecta es peor |
| E6 | ¿Converge el «yo pienso que él piensa»? | No: oscila (φ'(s\*) = −2); promediando (juego ficticio) sí |
| E7 | ¿Y sin suponer estrategias de umbral? | El PL exacto del juego discreto da umbrales que convergen a s\*, t\* |
| E8 | Juego ficticio | Converge al equilibrio |
| E9 | Monte Carlo de 10⁷ partidas | P(J1 gana) = 0,4864 ± 0,0003 (teoría 0,48624) |
| E10 | Análogos: anuncio secreto / sin complemento | 1/2 con umbral 1/√2 / ≈ 0,4943 con s³ + s = 3/4 |

## Pendiente

- Completar los integrantes del grupo en la portada (`entregable/plantilla.html.j2`, buscar `[completar]`) y volver a correr `python scripts/construir_entregable.py`.
