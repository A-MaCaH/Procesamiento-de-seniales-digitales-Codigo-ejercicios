"""
sistemas.py
===========
Sistemas discretos:  x[n] ---> [ SISTEMA ] ---> y[n]

1) Ecuación en diferencias (forma general de un sistema LTI causal):

       y[n] = b0·x[n] + b1·x[n-1] + ... + bM·x[n-M]
                      - a1·y[n-1] - ... - aK·y[n-K]

   b = [b0, b1, ..., bM]   coeficientes de la ENTRADA
   a = [1,  a1, ..., aK]   coeficientes de la SALIDA (a0 siempre es 1)

2) Respuesta al impulso h[n]: la salida cuando la entrada es δ[n].

3) Convolución: para un sistema LTI, la salida para CUALQUIER entrada es

       y[n] = x[n] * h[n] = Σ_k  x[k] · h[n - k]

   ¿Por qué? Porque x[n] es una suma de impulsos  Σ x[k]·δ[n-k];
   cada impulso δ[n-k] produce h[n-k] (invarianza), y por linealidad
   la salida es la suma  Σ x[k]·h[n-k].

   La función de convolución está en convolucion.py, junto con la
   convolución en 2D para imágenes.
"""

import numpy as np

__all__ = [
    "ecuacion_diferencias",
    "respuesta_al_impulso",
    "promedio_movil",
    "eco",
    "probar_linealidad",
    "probar_invarianza",
    "es_estable",
]


def ecuacion_diferencias(b, a, x):
    """
    Pasa la señal x por el sistema descrito por la ecuación en diferencias:

        y[n] = Σ_{k=0}^{M} b[k]·x[n-k]  -  Σ_{k=1}^{K} a[k]·y[n-k]

    Suponemos que el sistema empieza "en reposo": x[n] = 0 y y[n] = 0 para n < 0.

    Parámetros
    ----------
    b : lista  Coeficientes de la entrada [b0, b1, ...]
    a : lista  Coeficientes de la salida  [1, a1, a2, ...]  (usar [1] si no hay)
    x : arreglo  Señal de entrada (x[0] es la muestra n = 0)

    Ejemplo
    -------
    Promedio de 2 muestras:  y[n] = 0.5·x[n] + 0.5·x[n-1]
    >>> ecuacion_diferencias(b=[0.5, 0.5], a=[1], x=[2, 4, 6])
    array([1., 3., 5.])
    """
    x = np.asarray(x, dtype=float)
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)

    if a[0] != 1:
        raise ValueError("El primer coeficiente a[0] debe ser 1.")

    N = len(x)
    y = np.zeros(N)

    # Calculamos la salida muestra por muestra (n = 0, 1, 2, ...)
    for n in range(N):

        # Parte 1: suma de las entradas actuales y pasadas  Σ b[k]·x[n-k]
        suma_entrada = 0.0
        for k in range(len(b)):
            if n - k >= 0:                      # x[n-k] = 0 si n-k < 0
                suma_entrada = suma_entrada + b[k] * x[n - k]

        # Parte 2: suma de las salidas pasadas  Σ a[k]·y[n-k]  (desde k = 1)
        suma_salida = 0.0
        for k in range(1, len(a)):
            if n - k >= 0:                      # y[n-k] = 0 si n-k < 0
                suma_salida = suma_salida + a[k] * y[n - k]

        # Parte 3: la ecuación en diferencias
        y[n] = suma_entrada - suma_salida

    return y


def respuesta_al_impulso(b, a, N):
    """
    Respuesta al impulso h[n] del sistema (b, a): la salida cuando la
    entrada es el impulso δ[n].

    Pasos:
      1. Creamos δ[n] con N muestras: [1, 0, 0, ..., 0]
      2. Lo pasamos por el sistema.

    Regresa
    -------
    n, h : índices 0..N-1 y los valores de h[n].
    """
    # Paso 1: impulso de N muestras
    delta = np.zeros(N)
    delta[0] = 1.0

    # Paso 2: pasar por el sistema
    h = ecuacion_diferencias(b, a, delta)

    n = np.arange(0, N)
    return n, h


def promedio_movil(x, M):
    """
    Filtro de promedio móvil de M muestras (un filtro pasa-bajas sencillo):

        y[n] = (1/M) · ( x[n] + x[n-1] + ... + x[n-M+1] )

    Es la ecuación en diferencias con b = [1/M, 1/M, ..., 1/M] y a = [1].
    Suaviza la señal: quita variaciones rápidas (ruido de alta frecuencia).
    """
    b = np.ones(M) / M
    a = [1.0]
    y = ecuacion_diferencias(b, a, x)
    return y


def eco(x, retardo, alfa):
    """
    Agrega un eco a la señal:

        y[n] = x[n] + alfa · x[n - retardo]

    Parámetros
    ----------
    retardo : int    retraso del eco EN MUESTRAS (retardo_segundos · fs)
    alfa    : float  qué tan fuerte es el eco (0 < alfa < 1)

    Su respuesta al impulso es  h[n] = δ[n] + alfa·δ[n - retardo].

    (Es la ecuación en diferencias con b = [1, 0, 0, ..., 0, alfa], pero la
    programamos directamente porque casi todos los coeficientes son cero.)
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros(len(x))
    for n in range(len(x)):
        if n - retardo >= 0:
            y[n] = x[n] + alfa * x[n - retardo]
        else:
            y[n] = x[n]                 # todavía no llega el eco
    return y


def probar_linealidad(sistema, x1, x2, c1=2.0, c2=-3.0, tolerancia=1e-9):
    """
    Prueba (numérica) de LINEALIDAD de un sistema:

        sistema(c1·x1 + c2·x2)  ==  c1·sistema(x1) + c2·sistema(x2)  ?

    'sistema' es una función de Python que recibe un arreglo x y regresa y.
    Regresa True si la igualdad se cumple (para estas entradas).
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    # Lado izquierdo: primero combinamos, luego pasamos por el sistema
    izquierda = sistema(c1 * x1 + c2 * x2)

    # Lado derecho: primero pasamos por el sistema, luego combinamos
    derecha = c1 * sistema(x1) + c2 * sistema(x2)

    diferencia = np.max(np.abs(izquierda - derecha))
    print(f"Diferencia máxima entre ambos lados: {diferencia:.3g}")
    return bool(diferencia < tolerancia)


def probar_invarianza(sistema, x, k=3, tolerancia=1e-9):
    """
    Prueba (numérica) de INVARIANZA EN EL TIEMPO:

        Si  x[n] -> y[n],  entonces  x[n - k] -> y[n - k]  ?

    Pasos:
      1. y = sistema(x)
      2. Retrasamos la entrada k muestras (agregando k ceros al inicio) y la
         pasamos por el sistema.
      3. Retrasar la salida y k muestras significa que y[n] debe aparecer
         a partir de la posición k. Comparamos la salida del paso 2 desde
         la posición k en adelante con y. (Antes de k la señal original
         "todavía no existía", así que no hay nada que comparar.)
    """
    x = np.asarray(x, dtype=float)
    ceros = np.zeros(k)

    # Paso 1
    y = sistema(x)

    # Paso 2: entrada retrasada -> sistema
    x_retrasada = np.concatenate([ceros, x])
    salida_de_entrada_retrasada = sistema(x_retrasada)

    # Paso 3: comparar desde la posición k
    diferencia = np.max(np.abs(salida_de_entrada_retrasada[k:] - y))
    print(f"Diferencia máxima: {diferencia:.3g}")
    return bool(diferencia < tolerancia)


def es_estable(h, umbral=1e6):
    """
    Un sistema LTI es estable (BIBO) si su respuesta al impulso es
    absolutamente sumable:

        S = Σ |h[n]|  <  ∞

    Numéricamente sólo podemos ver un número finito de muestras de h, así
    que calculamos S y revisamos que no sea "enorme" y que la cola de h
    se haga pequeña.
    """
    h = np.asarray(h, dtype=float)
    S = 0.0
    for i in range(len(h)):
        S = S + abs(h[i])

    ultimas = np.abs(h[-10:])          # las últimas 10 muestras
    cola_pequena = bool(np.max(ultimas) < 1e-3 * max(np.max(np.abs(h)), 1e-12))
    print(f"Σ|h[n]| = {S:.4g}   (cola de h pequeña: {cola_pequena})")
    return bool(S < umbral and cola_pequena)
