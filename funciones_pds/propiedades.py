"""
propiedades.py
==============
Propiedades de una señal discreta: energía, potencia, valor RMS y periodo.

    Energía               E = Σ |x[n]|²
    Potencia promedio     P = (1/N) Σ |x[n]|²         (sobre N muestras)
    Potencia (periódica)  P = (1/N) Σ_{n=0}^{N-1} |x[n]|²   (sobre UN periodo N)
    Valor RMS             x_rms = √P

    * Señal de ENERGÍA:  0 < E < ∞     (P = 0)       ej. un pulso, un impulso
    * Señal de POTENCIA: 0 < P < ∞     (E = ∞)       ej. una senoidal, un escalón
"""

import numpy as np

__all__ = [
    "energia",
    "e_promedio",
    "potencia_periodica",
    "valor_rms",
    "es_periodica",
    "estimar_periodo",
]


def energia(x):
    """
    Energía de una señal discreta:

        E = Σ_n |x[n]|²

    Pasos: recorremos cada muestra, la elevamos al cuadrado y acumulamos.

    Ejemplo
    -------
    >>> energia([1, 2, 3])
    14.0                        # 1² + 2² + 3²
    """
    x = np.asarray(x)
    E = 0.0
    for i in range(len(x)):
        magnitud = abs(x[i])                 # |x[n]| (funciona también con complejos)
        E = E + magnitud ** 2                # acumulamos |x[n]|²
    return float(E)


def e_promedio(x):
    """
    Potencia promedio de las N muestras que tenemos:

        P = (1/N) Σ_n |x[n]|²  =  E / N

    Ejemplo
    -------
    >>> e_promedio([1, -1, 1, -1])
    1.0
    """
    x = np.asarray(x)
    N = len(x)
    E = energia(x)
    P = E / N
    return P


def potencia_periodica(x, N):
    """
    Potencia de una señal PERIÓDICA de periodo N: sólo hace falta
    promediar sobre UN periodo (las primeras N muestras).

        P = (1/N) Σ_{n=0}^{N-1} |x[n]|²

    Ejemplo
    -------
    Para una senoidal de amplitud A el resultado debe ser A²/2.
    """
    x = np.asarray(x)
    if len(x) < N:
        raise ValueError(f"La señal tiene {len(x)} muestras: no alcanza para un periodo de {N}.")

    un_periodo = x[0:N]                      # muestras n = 0 ... N-1
    P = energia(un_periodo) / N
    return P


def valor_rms(x):
    """
    Valor RMS (raíz cuadrática media) = raíz de la potencia promedio.

        x_rms = √( (1/N) Σ |x[n]|² )

    Para una senoidal de amplitud A:  x_rms = A / √2.
    """
    P = e_promedio(x)
    rms = np.sqrt(P)
    return float(rms)


def es_periodica(x, N, tolerancia=1e-6):
    """
    ¿Se repite la señal cada N muestras?   x[n + N] = x[n]  para todo n

    Pasos:
      1. Comparamos x[n] con x[n + N] para todas las n en las que ambas existen.
      2. Si la diferencia más grande es menor que la tolerancia -> sí es periódica con N.

    Nota: la tolerancia es necesaria porque las computadoras redondean
    (por ejemplo cos(2π) da 1.0000000000000002 en lugar de 1).
    """
    x = np.asarray(x)
    if N <= 0 or N >= len(x):
        return False

    diferencia_maxima = 0.0
    for i in range(len(x) - N):
        diferencia = abs(x[i + N] - x[i])
        if diferencia > diferencia_maxima:
            diferencia_maxima = diferencia

    return bool(diferencia_maxima < tolerancia)


def estimar_periodo(x, tolerancia=1e-6):
    """
    Busca el periodo fundamental de una señal: el número N MÁS PEQUEÑO
    para el que x[n + N] = x[n].

    Pasos:
      Probamos N = 1, 2, 3, ... (hasta la mitad de la señal, para poder
      comparar al menos un periodo completo) y nos quedamos con el primero
      que funcione. Si ninguno funciona regresamos None.

    Ojo: con señales REALES (con ruido) casi nunca se cumple exactamente;
    para ellas usaremos la AUTOCORRELACIÓN (notebook 07).
    """
    x = np.asarray(x)
    for N in range(1, len(x) // 2 + 1):
        if es_periodica(x, N, tolerancia):
            return N
    return None
