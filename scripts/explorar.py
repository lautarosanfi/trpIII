#!/usr/bin/env python3
"""Exploración por simulación: ¿qué estamos buscando?

Recorre, en orden, las hipótesis iniciales del grupo y las pone a prueba con
simulación Monte Carlo, cálculo numérico y el juego discretizado resuelto por PL.

Uso (desde la raíz del repo):
    python scripts/explorar.py            # corrida completa (~1 minuto)
    python scripts/explorar.py --rapido   # menos partidas, para probar cambios

Salida: resultados/resultados.json  (luego: python scripts/graficar.py)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402

from juego import analitico as A  # noqa: E402
from juego import discreto as disc  # noqa: E402
from juego import modelo as M  # noqa: E402
from juego import numerico as Nu  # noqa: E402
from juego.modelo import COMPLEMENTO, DESCARTAR, PLANTARSE  # noqa: E402

SEMILLA = 2026
S, T = A.S_ESTRELLA, A.T_ESTRELLA


def titulo(texto: str) -> None:
    print(f"\n{'=' * 78}\n{texto}\n{'=' * 78}")


def est(e: M.Estimacion) -> dict:
    return {"p": e.p, "n": e.n, "ee": e.error_estandar, "ic95": list(e.ic95)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rapido", action="store_true", help="usar menos partidas (para probar)")
    args = ap.parse_args()
    f = 0.1 if args.rapido else 1.0
    n_grande, n_medio, n_chico = int(10_000_000 * f), int(2_000_000 * f), int(200_000 * f)

    t0 = time.time()
    rng = np.random.default_rng(SEMILLA)
    R: dict = {"semilla": SEMILLA, "rapido": args.rapido, "analitico": A.resumen()}

    # ------------------------------------------------------------------ E1
    titulo("E1. Hipótesis inicial: complementar si x < 1/2, plantarse si no (nunca descartar)")
    ingenuo = M.j1_umbral(0.5)
    a1, v1 = M.simular_j1(ingenuo, n_medio, rng)
    media_v1 = float(v1.mean())
    print(f"  E[número final de J1] simulado = {media_v1:.4f}   (teoría: E[max(X,1-X)] = 3/4)")
    vs_ingenuo = M.estimar_prob_j1(ingenuo, M.j2_umbrales(0.5, 0.5), n_medio, rng)
    print(f"  contra un J2 que hace lo mismo:  P(J1 gana) = {vs_ingenuo}   (teoría 1/2)")
    a1, v1 = M.simular_j1(ingenuo, n_chico, rng)
    e2_aprendido, umb = Nu.mejor_respuesta_empirica(a1, v1)
    print(f"  J2 observa {n_chico:,} partidas de este J1 y aprende sus umbrales:")
    print(f"     tras 'plantarse' {umb[PLANTARSE]:.4f}, tras 'complemento' {umb[COMPLEMENTO]:.4f}"
          f"   (teoría t(1/2) = 5/8 = 0.625)")
    vs_aprendido = M.estimar_prob_j1(ingenuo, e2_aprendido, n_medio, rng)
    print(f"  contra ese J2:  P(J1 gana) = {vs_aprendido}   (teoría g(1/2) = 15/32 = {15 / 32:.5f})")
    R["e1"] = {"media_v1": media_v1, "vs_j2_ingenuo": est(vs_ingenuo),
               "umbrales_aprendidos": {str(k): v for k, v in umb.items()},
               "vs_j2_aprendido": est(vs_aprendido), "teoria_vs_j2_optimo": 15 / 32}

    # ------------------------------------------------------------------ E2
    titulo("E2. ¿Conviene descartar números medianos? Barrido del umbral s de J1 (J2 aprende)")
    grilla_s = np.round(np.arange(0.50, 0.8001, 0.01), 2)
    filas = []
    for k, s in enumerate(grilla_s):
        e1 = M.j1_umbral(s)
        a1, v1 = M.simular_j1(e1, n_chico, np.random.default_rng(1000 + k))
        e2, u = Nu.mejor_respuesta_empirica(a1, v1)
        # misma semilla para todos los s (números aleatorios comunes): la curva sale más suave
        p = M.estimar_prob_j1(e1, e2, n_medio, np.random.default_rng(77))
        filas.append({"s": float(s), "umbral_p": u[PLANTARSE], "umbral_c": u[COMPLEMENTO],
                      "umbral_d": u[DESCARTAR], "prob": est(p), "prob_teoria": float(A.prob_j1_umbral(s)),
                      "umbral_teoria": float(A.umbral_j2(s))})
    mejor = max(filas, key=lambda r: r["prob"]["p"])
    for r in filas[::5]:
        print(f"  s = {r['s']:.2f}: J2 aprende t = {r['umbral_p']:.3f}/{r['umbral_c']:.3f} (teoría {r['umbral_teoria']:.3f}),"
              f" tras descarte {r['umbral_d']:.3f};  P(J1 gana) = {r['prob']['p']:.4f} (teoría {r['prob_teoria']:.4f})")
    print(f"  máximo simulado en s = {mejor['s']:.2f} (P = {mejor['prob']['p']:.4f});"
          f"  máximo teórico en s* = 4^(-1/3) = {S:.4f} (P = {A.P_GANA_J1:.4f})")
    # Con números aleatorios comunes todos los puntos comparten UN error aleatorio: la curva
    # simulada es suave pero puede quedar corrida en bloque. ¿Es azar o hay un sesgo?
    # Se repite la evaluación en s* con otras 12 semillas.
    n_rep = n_medio
    reps = [M.estimar_prob_j1(M.j1_umbral(S), M.j2_umbrales(T, 0.5), n_rep, np.random.default_rng(78 + i)).p
            for i in range(12)]
    ee = float(np.sqrt(A.P_GANA_J1 * (1 - A.P_GANA_J1) / n_rep))
    p77 = M.estimar_prob_j1(M.j1_umbral(S), M.j2_umbrales(T, 0.5), n_rep, np.random.default_rng(77)).p
    print(f"  ¿sesgo o azar? en s* con la semilla de la curva: {p77:.5f} (z = {(p77 - A.P_GANA_J1) / ee:+.2f});"
          f"  12 semillas más: media {np.mean(reps):.5f}, desvío {np.std(reps, ddof=1):.5f} (EE teórico {ee:.5f})")
    R["e2"] = {"filas": filas, "s_mejor_simulado": mejor["s"],
               "chequeo_semillas": {"p_semilla_curva": p77, "z_semilla_curva": (p77 - A.P_GANA_J1) / ee,
                                    "otras": reps, "media_otras": float(np.mean(reps)),
                                    "desvio_otras": float(np.std(reps, ddof=1)), "ee_teorico": ee, "n": n_rep}}

    # ------------------------------------------------------------------ E3
    titulo("E3. ¿A qué número apunta J2 cuando J1 se planta? (J1 juega con umbral s*)")
    media_plantado = (1 + S) / 2
    candidatos = [
        ("siempre se planta (no usa el anuncio)", 0.5),
        ("5/8 = 0,625 (hipótesis)", 0.625),
        ("3/4 = 0,75 (hipótesis)", 0.75),
        ("E[V1 | plantó] = (1+s*)/2 (superar el promedio)", media_plantado),
        ("t* = (1+s*²)/2 (óptimo)", T),
    ]
    tabla_j2 = []
    for nombre, t in candidatos:
        cond = float(A.prob_j2_tras_plantarse(S, t))
        total = 1 - float(A.prob_j1(S, t, 0.5))
        mc = M.estimar_prob_j1(M.j1_umbral(S), M.j2_umbrales(t, 0.5), n_medio, rng)
        tabla_j2.append({"nombre": nombre, "t": t, "p_j2_si_planto": cond, "p_j2_total": total,
                         "p_j2_total_mc": 1 - mc.p, "ee_mc": mc.error_estandar})
        print(f"  umbral {t:.4f}  {nombre:<48s} P(J2 gana | J1 plantó) = {cond:.4f};"
              f"  P(J2 gana) = {total:.4f}  (MC {1 - mc.p:.4f})")
    R["e3"] = {"tabla": tabla_j2, "media_v1_si_planto": media_plantado,
               "p_descartar_si_planto": (1 - S) / 2}

    # ------------------------------------------------------------------ E4
    titulo("E4. Un farol puntual: J1 con un número flojo, ¿se planta o descarta? (J2 juega t*)")
    e2_eq = M.j2_umbrales(T, 0.5)
    tabla_j1 = []
    for m in [0.52, 0.55, 0.60, S, 0.65, 0.70, 0.80, 0.90]:
        pl = float(A.pago_plantarse(m, T))
        # verificación Monte Carlo: J2 ve "plantarse" y responde con t*; ¿qué tan seguido V2 < m?
        z, w = rng.random((2, n_medio // 2))
        a2 = e2_eq.actuar(np.full(len(z), PLANTARSE), z, rng)
        mc = float((M.numero_final(a2, z, w) < m).mean())
        tabla_j1.append({"m": float(m), "plantarse": pl, "plantarse_mc": mc, "descartar": 0.25})
        print(f"  m = {m:.4f}: plantarse gana {pl:.4f} (MC {mc:.4f})  vs  descartar 0.2500"
              f"  -> {'plantarse' if pl > 0.25 else 'descartar'}")
    R["e4"] = {"tabla": tabla_j1}

    # ------------------------------------------------------------------ E5
    titulo("E5. Faroles sistemáticos: ¿puede J1 ganar algo engañando? (cálculo numérico exacto)")
    qs = np.linspace(0, 1, 21)
    farol_q = []
    for q in qs:
        e1 = M.j1_farol_aleatorio(S, q)
        _, u = Nu.mejor_respuesta_j2(e1)
        farol_q.append({"q": float(q), "vs_mejor_respuesta": Nu.prob_j1_contra_mejor_respuesta(e1),
                        "vs_t_estrella": Nu.prob_j1_contra_umbrales(e1, T, T, 0.5),
                        "umbral_j2": u[PLANTARSE]})
    bs = np.linspace(0.5, S, 21)
    farol_b = []
    for b in bs:
        e1 = M.j1_farol_polarizado(S, b)
        _, u = Nu.mejor_respuesta_j2(e1)
        farol_b.append({"b": float(b), "vs_mejor_respuesta": Nu.prob_j1_contra_mejor_respuesta(e1),
                        "vs_t_estrella": Nu.prob_j1_contra_umbrales(e1, T, T, 0.5),
                        "umbral_j2": u[PLANTARSE]})
    for r in farol_q[::5]:
        print(f"  farol con prob. q = {r['q']:.2f}: J2 se adapta -> {r['vs_mejor_respuesta']:.4f}"
              f" (umbral J2 {r['umbral_j2']:.3f});  J2 no se adapta -> {r['vs_t_estrella']:.4f}")
    for r in farol_b[::10]:
        print(f"  farol con las peores manos (max < {r['b']:.3f}): J2 se adapta -> {r['vs_mejor_respuesta']:.4f};"
              f"  J2 no se adapta -> {r['vs_t_estrella']:.4f}")
    # ¿y si J1 reacomoda también el umbral? mejor s para cada intensidad de farol
    grilla_fina = np.round(np.arange(0.55, 0.7001, 0.0025), 4)
    mejor_por_q = []
    for q in [0.0, 0.1, 0.2, 0.3, 0.5]:
        vals = [Nu.prob_j1_contra_mejor_respuesta(M.j1_farol_aleatorio(s, q), k=8000) for s in grilla_fina]
        i = int(np.argmax(vals))
        mejor_por_q.append({"q": q, "s_optimo": float(grilla_fina[i]), "prob": float(vals[i])})
        print(f"  con farol q = {q:.1f}, el mejor umbral es s = {grilla_fina[i]:.4f} y J1 gana {vals[i]:.4f}")
    # «plantarse» y «complemento» con umbrales distintos (para que el anuncio diga algo distinto)
    asim = []
    for sp, sc in [(S, S), (0.60, 0.66), (0.66, 0.60), (0.55, 0.72), (0.70, 0.70)]:
        v = Nu.prob_j1_contra_mejor_respuesta(M.j1_umbrales_por_anuncio(sp, sc))
        teo = float((A.prob_j1_umbral(sp) + A.prob_j1_umbral(sc)) / 2)
        asim.append({"s_p": float(sp), "s_c": float(sc), "prob": v, "teoria": teo})
        print(f"  umbrales distintos P:{sp:.3f} C:{sc:.3f} -> {v:.4f}  (teoría (g(sP)+g(sC))/2 = {teo:.4f})")
    # chequeo Monte Carlo de un farol, con J2 aprendiendo de los datos
    e1 = M.j1_farol_aleatorio(S, 0.3)
    a1, v1 = M.simular_j1(e1, n_chico * 2, rng)
    e2, u = Nu.mejor_respuesta_empirica(a1, v1)
    mc = M.estimar_prob_j1(e1, e2, n_medio, rng)
    num = Nu.prob_j1_contra_mejor_respuesta(e1)
    print(f"  MC farol q = 0,3 con J2 que aprende: {mc}   (numérico {num:.4f})")
    R["e5"] = {"aleatorio": farol_q, "polarizado": farol_b, "mejor_s_por_q": mejor_por_q,
               "asimetrico": asim, "mc_farol_03": {"mc": est(mc), "numerico": num,
                                                   "umbrales_aprendidos": {str(k): v for k, v in u.items()}}}

    # ------------------------------------------------------------------ E6
    titulo("E6. «Yo pienso que él piensa…»: mejores respuestas encadenadas (sin promediar)")
    s_seq, t_seq = A.iterar_mejores_respuestas(0.5, 24)
    print("  s_k:", " ".join(f"{v:.3f}" for v in s_seq[:12]), "…")
    s_cerca, _ = A.iterar_mejores_respuestas(S + 1e-3, 24)
    print("  arrancando en s* + 0,001:", " ".join(f"{v:.4f}" for v in s_cerca[:10]), "…")
    print(f"  derivada de la composición en s*: -1/(2 s*³) = {-1 / (2 * S**3):.3f}  (|·| > 1: punto fijo inestable)")
    R["e6"] = {"s": s_seq.tolist(), "t": t_seq.tolist(), "s_cerca": s_cerca.tolist()}

    # ------------------------------------------------------------------ E7
    titulo("E7. Juego discretizado resuelto exactamente por programación lineal (sin supuestos)")
    convergencia = []
    for n in [10, 20, 40, 80, 160, 320]:
        sol = disc.resolver(n)
        convergencia.append({"n": n, "valor": sol.valor, "valor_j2": sol.valor_j2,
                             "descarte_j1": sol.region_descarte_j1(),
                             "descarte_j2_tras_p": sol.region_descarte_j2(0),
                             "descarte_j2_tras_c": sol.region_descarte_j2(1),
                             "descarte_j2_tras_d": sol.region_descarte_j2(2),
                             "celdas_mixtas_j1": sol.celdas_mixtas_j1()})
        print(f"  N = {n:4d}: valor {sol.valor:.5f} (J2: {sol.valor_j2:.5f});  J1 descarta x en "
              f"{tuple(round(v, 3) for v in sol.region_descarte_j1())};  J2 tras P descarta z en "
              f"{tuple(round(v, 3) for v in sol.region_descarte_j2(0))};  tras D: {sol.region_descarte_j2(2)}")
    sol = disc.resolver(400)
    print(f"  N = 400: valor {sol.valor:.5f};  teoría (continuo) {A.P_GANA_J1:.5f}")
    R["e7"] = {"convergencia": convergencia, "n": 400, "valor": sol.valor,
               "estrategia_j1": sol.estrategia_j1.round(6).tolist(),
               "estrategia_j2": sol.estrategia_j2.round(6).tolist()}

    # ------------------------------------------------------------------ E8
    titulo("E8. Juego ficticio: cada uno responde al promedio de lo que vio del otro")
    fp = disc.juego_ficticio(100, 20_000 if not args.rapido else 2_000)
    print(f"  brecha (explotabilidad) tras {fp['iteracion'][-1]} rondas: {fp['brecha'][-1]:.2e};"
          f"  valor entre {fp['valor_inf'][-1]:.4f} y {fp['valor_sup'][-1]:.4f}")
    R["e8"] = {k: np.asarray(v).round(7).tolist() for k, v in fp.items()}

    # ------------------------------------------------------------------ E9
    titulo("E9. Monte Carlo del equilibrio (σ1*, σ2*): n grande y tabla condicional")
    e1, e2 = M.j1_umbral(S), M.j2_umbrales(T, 0.5)
    lote = 1_000_000 if not args.rapido else 200_000
    cuentas = {"n": 0, "gana": 0, "j1_planta": 0, "gana_si_planta": 0, "j1_descarta": 0,
               "gana_si_descarta": 0, "j2_descarta_si_planta": 0, "j2_descarta_si_descarta": 0}
    trayectoria = None
    while cuentas["n"] < n_grande:
        pa = M.simular(e1, e2, min(lote, n_grande - cuentas["n"]), rng)
        g = pa.gana_j1
        planta = pa.a1 != DESCARTAR
        if trayectoria is None:  # promedio acumulado del primer lote (ley de los grandes números)
            idx = np.unique(np.geomspace(10, len(g), 80).astype(int))
            trayectoria = {"n": idx.tolist(), "p": (np.cumsum(g)[idx - 1] / idx).tolist()}
        cuentas["n"] += len(g)
        cuentas["gana"] += int(g.sum())
        cuentas["j1_planta"] += int(planta.sum())
        cuentas["gana_si_planta"] += int(g[planta].sum())
        cuentas["j1_descarta"] += int((~planta).sum())
        cuentas["gana_si_descarta"] += int(g[~planta].sum())
        cuentas["j2_descarta_si_planta"] += int((pa.a2[planta] == DESCARTAR).sum())
        cuentas["j2_descarta_si_descarta"] += int((pa.a2[~planta] == DESCARTAR).sum())
    c = cuentas
    p_total = M.Estimacion(c["gana"] / c["n"], c["n"])
    an = A.resumen()
    mc = {
        "p_gana_j1": est(p_total),
        "p_j1_planta": c["j1_planta"] / c["n"],
        "p_gana_j1_si_planto": c["gana_si_planta"] / c["j1_planta"],
        "p_gana_j1_si_descarto": c["gana_si_descarta"] / c["j1_descarta"],
        "p_j2_descarta_si_j1_planto": c["j2_descarta_si_planta"] / c["j1_planta"],
        "p_j2_descarta_si_j1_descarto": c["j2_descarta_si_descarta"] / c["j1_descarta"],
    }
    print(f"  {c['n']:,} partidas: P(J1 gana) = {p_total}   (teoría {A.P_GANA_J1:.5f})")
    for clave, teo in [("p_j1_planta", an["p_j1_planta"]), ("p_gana_j1_si_planto", an["p_gana_j1_si_planto"]),
                       ("p_gana_j1_si_descarto", 0.25), ("p_j2_descarta_si_j1_planto", an["p_j2_descarta_si_j1_planto"]),
                       ("p_j2_descarta_si_j1_descarto", 0.0)]:
        print(f"    {clave:<32s} simulado {mc[clave]:.4f}   teoría {teo:.4f}")
    R["e9"] = {"mc": mc, "trayectoria": trayectoria}

    # ------------------------------------------------------------------ E10
    titulo("E10. Problemas análogos")
    # A) anuncio secreto
    ts = np.linspace(0.5, 1, 2001)
    curva_oculto = [float(np.min(A.prob_j1_oculto(s, ts))) for s in grilla_s]
    sol_o = disc.resolver(200, "anuncio_oculto")
    mc_o = M.estimar_prob_j1(M.j1_umbral(A.UMBRAL_OCULTO), M.j2_ignora_anuncio(A.UMBRAL_OCULTO),
                             n_medio, rng, ve_anuncio=False)
    print(f"  A) anuncio secreto: umbral 1/√2 = {A.UMBRAL_OCULTO:.4f}, P(J1) = 1/2;  PL N=200: {sol_o.valor:.5f};"
          f"  MC: {mc_o}")
    # B) sin complemento
    sb, tb = A.S_SIN_COMPLEMENTO, A.T_SIN_COMPLEMENTO
    sol_b = disc.resolver(400, "sin_complemento")
    e1b = M.j1_sin_complemento(sb)
    mc_b = M.estimar_prob_j1(e1b, M.j2_sin_complemento(tb, 0.5), n_medio, rng)
    a1, v1 = M.simular_j1(e1b, n_chico, rng)
    u_b = Nu.umbrales_empiricos(a1, v1)  # (el cuantil no depende de poder complementar)
    print(f"  B) sin complemento: s = {sb:.4f}, t = {tb:.4f}, P(J1) = {A.P_GANA_J1_SIN_COMPLEMENTO:.5f};"
          f"  PL N=400: {sol_b.valor:.5f};  MC: {mc_b}")
    R["e10"] = {
        "oculto": {"umbral": A.UMBRAL_OCULTO, "valor": 0.5, "curva_s": grilla_s.tolist(), "curva": curva_oculto,
                   "pl_valor": sol_o.valor, "pl_descarte_j1": sol_o.region_descarte_j1(),
                   "pl_descarte_j2": sol_o.region_descarte_j2(0), "mc": est(mc_o)},
        "sin_complemento": {"s": sb, "t": tb, "valor": A.P_GANA_J1_SIN_COMPLEMENTO,
                            "pl_valor": sol_b.valor, "pl_descarte_j1": sol_b.region_descarte_j1(),
                            "pl_descarte_j2_tras_p": sol_b.region_descarte_j2(0),
                            "pl_descarte_j2_tras_d": sol_b.region_descarte_j2(2),
                            "mc": est(mc_b), "umbrales_aprendidos": {str(k): v for k, v in u_b.items()},
                            "curva": A.prob_j1_umbral_sin_complemento(grilla_s).tolist()},
    }

    salida = RAIZ / "resultados" / "resultados.json"
    salida.parent.mkdir(exist_ok=True)
    salida.write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float))
    print(f"\nListo en {time.time() - t0:.0f} s -> {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
