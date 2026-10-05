"""
operaciones.py
==============
Operaciones entre señales discretas, RESPETANDO LOS ÍNDICES.

Problema típico: queremos sumar
    x1 definida en n = -2 ... 2
    x2 definida en n =  0 ... 5
No podemos sumar los arreglos "tal cual" (tienen distinto tamaño y además
x1[0] de Python NO es el mismo instante que x2[0]). Primero hay que
ALINEARLAS sobre un eje de índices común, rellenando con ceros.
"""

import numpy as np

__all__ = [
    "alinear",
    "sumar_senales",
    "multiplicar_senales",
    "escalar_amplitud",
    "desplazar",
    "invertir_tiempo",
    "submuestrear",
    "sobremuestrear",
    "parte_par_impar",
]


def alinear(n1, x1, n2, x2):
    """
    Lleva dos señales a un MISMO eje de índices (rellenando con ceros).

    Pasos:
      1. El nuevo eje va desde el índice más pequeño de las dos señales
         hasta el más grande.
      2. Creamos dos señales de ceros con ese tamaño.
      3. Copiamos cada valor de x1 (y de x2) en la posición que le corresponde.

    Regresa
    -------
    n, y1, y2 : eje común y las dos señales ya alineadas.
    """
    # Paso 1: eje común
    n_min = min(n1[0], n2[0])
    n_max = max(n1[-1], n2[-1])
    n = np.arange(n_min, n_max + 1)

    # Paso 2: señales de ceros
    y1 = np.zeros(len(n))
    y2 = np.zeros(len(n))

    # Paso 3: copiar cada muestra en su lugar.
    # El índice n1[i] está en la posición (n1[i] - n_min) del nuevo arreglo.
    for i in range(len(n1)):
        posicion = n1[i] - n_min
        y1[posicion] = x1[i]

    for i in range(len(n2)):
        posicion = n2[i] - n_min
        y2[posicion] = x2[i]

    return n, y1, y2


def sumar_senales(n1, x1, n2, x2):
    """
    y[n] = x1[n] + x2[n]   (muestra a muestra, en el MISMO índice n)
    """
    n, y1, y2 = alinear(n1, x1, n2, x2)
    y = y1 + y2
    return n, y


def multiplicar_senales(n1, x1, n2, x2):
    """
    y[n] = x1[n] · x2[n]   (muestra a muestra, en el MISMO índice n)

    Uso típico: "ventanear" o "recortar" una señal multiplicándola por un pulso.
    """
    n, y1, y2 = alinear(n1, x1, n2, x2)
    y = y1 * y2
    return n, y


def escalar_amplitud(n, x, a):
    """
    y[n] = a · x[n]    (amplifica si |a| > 1, atenúa si |a| < 1, invierte si a < 0)

    Los índices NO cambian.
    """
    y = a * np.asarray(x)
    return np.array(n), y


def desplazar(n, x, k):
    """
    Desplazamiento en el tiempo:   y[n] = x[n - k]

        k > 0  -> la señal se RETRASA (se mueve a la DERECHA)
        k < 0  -> la señal se ADELANTA (se mueve a la IZQUIERDA)

    Los valores no cambian; sólo cambian los índices. Cada valor que estaba
    en el índice m ahora está en el índice m + k.
    """
    n_nuevo = np.array(n) + k
    y = np.array(x)
    return n_nuevo, y


def invertir_tiempo(n, x):
    """
    Inversión (reflexión) en el tiempo:   y[n] = x[-n]

    Pasos:
      1. Cada índice m pasa a ser -m.
      2. Como el eje queda "al revés" (de mayor a menor), invertimos el
         orden de los dos arreglos para que n vuelva a ir de menor a mayor.
    """
    # Paso 1: cambiar el signo de los índices
    n_reflejado = -np.array(n)

    # Paso 2: invertir el orden ([::-1] recorre el arreglo de atrás hacia adelante)
    n_nuevo = n_reflejado[::-1]
    y = np.array(x)[::-1]
    return n_nuevo, y


def submuestrear(n, x, M):
    """
    Submuestreo (diezmado) por un factor entero M:   y[n] = x[M·n]

    Nos quedamos con una de cada M muestras (las que tienen índice
    múltiplo de M). La señal se "comprime" en el tiempo.
    Si x tenía frecuencia de muestreo fs, y tiene fs / M.
    """
    n_nuevo = []
    y = []
    for i in range(len(n)):
        if n[i] % M == 0:                # ¿el índice es múltiplo de M?
            n_nuevo.append(n[i] // M)    # el índice M·m pasa a ser m
            y.append(x[i])
    return np.array(n_nuevo), np.array(y, dtype=float)


def sobremuestrear(n, x, L):
    """
    Sobremuestreo (expansión) por un factor entero L:

        y[n] = x[n / L]   si n es múltiplo de L
               0          en otro caso

    Se insertan L-1 ceros entre cada par de muestras: la señal se
    "estira" en el tiempo.
    """
    n_nuevo = np.arange(n[0] * L, n[-1] * L + 1)
    y = np.zeros(len(n_nuevo))
    for i in range(len(n)):
        posicion = n[i] * L - n_nuevo[0]
        y[posicion] = x[i]
    return n_nuevo, y


def parte_par_impar(n, x):
    """
    Toda señal se puede escribir como suma de una parte PAR y una IMPAR:

        x_par[n]   = ( x[n] + x[-n] ) / 2
        x_impar[n] = ( x[n] - x[-n] ) / 2
        x[n] = x_par[n] + x_impar[n]

    Regresa
    -------
    n_comun, x_par, x_impar
    """
    # x[-n]
    n_inv, x_inv = invertir_tiempo(n, x)

    # alinear x[n] y x[-n] en el mismo eje
    n_comun, a, b = alinear(n, x, n_inv, x_inv)

    x_par = (a + b) / 2
    x_impar = (a - b) / 2
    return n_comun, x_par, x_impar
