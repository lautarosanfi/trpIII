"""Herramientas para analizar «El Juego de los Números Uniformes» (TRP III, 2026).

Módulos
-------
modelo     reglas del juego, estrategias y simulador Monte Carlo.
analitico  fórmulas cerradas (equilibrio, mejores respuestas, variantes).
numerico   mejor respuesta de J2 calculada numéricamente para cualquier estrategia
           de J1 (integración en grilla fina) o aprendida de partidas simuladas.
discreto   versión discreta del juego resuelta exactamente por programación lineal,
           sin suponer la forma de las estrategias; juego ficticio.
figuras    estilo y funciones de graficación.
"""
