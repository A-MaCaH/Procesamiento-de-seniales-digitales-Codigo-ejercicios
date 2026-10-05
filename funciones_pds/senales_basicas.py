"""
senales_basicas.py
==================
Funciones para CREAR señales discretas.

CONVENCIÓN DEL CURSO (muy importante)
------------------------------------
Una señal discreta NO es sólo una lista de valores: también necesitamos
saber en QUÉ ÍNDICE n está cada valor. Por eso casi todas las funciones
regresan DOS arreglos del mismo tamaño:

    n : los índices de tiempo     ->  [-3, -2, -1,  0,  1,  2,  3]
    x : los valores de la señal   ->  [ 0,  0,  0,  1,  0,  0,  0]

    x[0] de Python es el PRIMER elemento del arreglo, que corresponde
    al índice n[0] = -3, no al instante n = 0.
"""

import numpy as np

__all__ = [
    "indices",
    "impulso",
    "escalon",
    "rampa",
    "exponencial",
    "senoidal_digital_muestras",
    "senoidal_digital",
    "descomponer_en_impulsos",
    "reconstruir_desde_impulsos",
]


def indices(n_inicio, n_fin):
    """
    Crea el vector de índices enteros n = n_inicio, n_inicio+1, ..., n_fin
    (INCLUYENDO n_fin).

    Ejemplo
    -------
    >>> indices(-2, 3)
    array([-2, -1,  0,  1,  2,  3])
    """
    # np.arange NO incluye el último valor, por eso sumamos 1
    n = np.arange(n_inicio, n_fin + 1)
    return n


def impulso(n_inicio, n_fin, n0=0):
    """
    Impulso unitario (delta de Kronecker) desplazado a n0:

        δ[n - n0] = 1  si n = n0
                    0  en cualquier otro caso

    Parámetros
    ----------
    n_inicio, n_fin : int   Primer y último índice que queremos ver.
    n0              : int   Posición del impulso (por defecto 0).

    Regresa
    -------
    n, x : arreglos con los índices y los valores.

    Ejemplo
    -------
    >>> n, x = impulso(-3, 3)
    >>> x
    array([0., 0., 0., 1., 0., 0., 0.])
    """
    # Paso 1: vector de índices
    n = indices(n_inicio, n_fin)

    # Paso 2: empezamos con una señal llena de ceros (mismo tamaño que n)
    x = np.zeros(len(n))

    # Paso 3: recorremos cada índice y ponemos 1 donde n vale n0
    for i in range(len(n)):
        if n[i] == n0:
            x[i] = 1.0

    return n, x


def escalon(n_inicio, n_fin, n0=0):
    """
    Escalón unitario desplazado a n0:

        u[n - n0] = 1  si n >= n0
                    0  si n <  n0

    Ejemplo
    -------
    >>> n, x = escalon(-3, 3)
    >>> x
    array([0., 0., 0., 1., 1., 1., 1.])
    """
    # Paso 1: vector de índices
    n = indices(n_inicio, n_fin)

    # Paso 2: señal de ceros
    x = np.zeros(len(n))

    # Paso 3: ponemos 1 en todos los índices mayores o iguales a n0
    for i in range(len(n)):
        if n[i] >= n0:
            x[i] = 1.0

    return n, x


def rampa(n_inicio, n_fin, n0=0):
    """
    Rampa unitaria desplazada a n0:

        r[n - n0] = (n - n0)  si n >= n0
                     0        si n <  n0

    Ejemplo
    -------
    >>> n, x = rampa(-2, 4)
    >>> x
    array([0., 0., 0., 1., 2., 3., 4.])
    """
    n = indices(n_inicio, n_fin)
    x = np.zeros(len(n))

    for i in range(len(n)):
        if n[i] >= n0:
            x[i] = n[i] - n0

    return n, x


def exponencial(n_inicio, n_fin, a, A=1.0):
    """
    Exponencial discreta:

        x[n] = A · a^n

        |a| < 1  -> decrece          a > 0 -> siempre del mismo signo
        |a| > 1  -> crece            a < 0 -> alterna el signo

    Ejemplo
    -------
    >>> n, x = exponencial(0, 4, a=0.5)
    >>> x
    array([1.    , 0.5   , 0.25  , 0.125 , 0.0625])
    """
    n = indices(n_inicio, n_fin)

    # numpy eleva 'a' a cada uno de los valores de n (uno por uno)
    x = A * (a ** n.astype(float))

    return n, x


def senoidal_digital_muestras(N, f, A=1.0, fase=0.0):
    """
    Senoidal DIGITAL definida directamente con la frecuencia digital f:

        x[n] = A · cos(2π f n + fase),     n = 0, 1, ..., N-1

    Aquí NO aparece ningún tiempo en segundos: sólo índices.

    Parámetros
    ----------
    N    : int    Número de muestras.
    f    : float  Frecuencia digital en ciclos/muestra (normalmente 0 <= f <= 0.5).
    A    : float  Amplitud.
    fase : float  Fase en radianes.

    Ejemplo
    -------
    >>> n, x = senoidal_digital_muestras(N=16, f=1/8)   # 8 muestras por ciclo -> 2 ciclos
    """
    # Paso 1: índices n = 0, 1, ..., N-1
    n = np.arange(0, N)

    # Paso 2: frecuencia angular digital ω = 2πf  [rad/muestra]
    w = 2 * np.pi * f

    # Paso 3: evaluamos la fórmula (numpy la evalúa para cada n)
    x = A * np.cos(w * n + fase)

    return n, x


def senoidal_digital(F, fs, duracion, A=1.0, fase=0.0):
    """
    Senoidal obtenida al MUESTREAR una senoidal analógica de F Hz
    con frecuencia de muestreo fs durante 'duracion' segundos:

        x(t) = A · cos(2π F t + fase)
        t_n  = n / fs
        x[n] = x(t_n) = A · cos(2π (F/fs) n + fase)

    Regresa
    -------
    n : índices de las muestras  (0, 1, 2, ...)
    t : instante en segundos de cada muestra (t = n / fs)
    x : valores de la señal

    Ejemplo
    -------
    >>> n, t, x = senoidal_digital(F=440, fs=8000, duracion=1.0)   # un "La" de 1 s
    """
    # Paso 1: ¿cuántas muestras caben en la duración?  N = duracion · fs
    N = int(round(duracion * fs))

    # Paso 2: índices de las muestras
    n = np.arange(0, N)

    # Paso 3: instante (en segundos) de cada muestra
    Ts = 1 / fs
    t = n * Ts

    # Paso 4: evaluamos la senoidal analógica en esos instantes
    x = A * np.cos(2 * np.pi * F * t + fase)

    return n, t, x



def descomponer_en_impulsos(n, x):
    """
    "Toda señal discreta es una suma de impulsos desplazados y escalados":

        x[n] = Σ_k  x[k] · δ[n - k]

    Esta función regresa una lista con cada uno de esos impulsos
    escalados: el término  x[k] · δ[n - k]  para cada k.

    Regresa
    -------
    componentes : lista de arreglos (uno por cada muestra de x).
    """
    componentes = []

    # Recorremos cada muestra de la señal
    for i in range(len(n)):
        k = n[i]                    # posición (índice) de esta muestra
        valor = x[i]                # valor de la señal en esa posición

        # impulso ubicado en k, con el MISMO eje de índices que x
        _, delta_k = impulso(n[0], n[-1], n0=k)

        # lo escalamos por el valor de la señal
        termino = valor * delta_k
        componentes.append(termino)

    return componentes


def reconstruir_desde_impulsos(componentes):
    """
    Suma todos los impulsos escalados de 'descomponer_en_impulsos'
    para volver a obtener la señal original.
    """
    # Empezamos con una señal de ceros del tamaño de los componentes
    y = np.zeros(len(componentes[0]))

    # Sumamos uno por uno
    for termino in componentes:
        y = y + termino

    return y
