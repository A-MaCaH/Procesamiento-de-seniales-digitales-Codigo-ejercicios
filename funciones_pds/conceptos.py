"""
conceptos.py
============
Funciones pequeñas que convierten entre las magnitudes de una señal
ANALÓGICA (tiempo continuo) y una señal DIGITAL (tiempo discreto).

Tabla de símbolos que usamos en todo el curso
---------------------------------------------
    Señal analógica  x(t) = A cos(Ω t + φ) = A cos(2π F t + φ)
        F   : frecuencia analógica ............ [Hz] = [ciclos / segundo]
        Ω   : velocidad (frecuencia) angular .. [rad / segundo]      Ω = 2π F
        T   : periodo ......................... [segundos]           T = 1 / F

    Muestreo
        fs  : frecuencia de muestreo .......... [muestras / segundo] (Hz)
        Ts  : periodo de muestreo ............. [segundos]           Ts = 1 / fs
        t_n : instante de la muestra n ........ [segundos]           t_n = n · Ts = n / fs

    Señal digital    x[n] = A cos(ω n + φ) = A cos(2π f n + φ)
        f   : frecuencia digital .............. [ciclos / muestra]   f = F / fs
        ω   : frecuencia angular digital ...... [rad / muestra]      ω = 2π f = Ω / fs
        N   : periodo fundamental ............. [muestras]          (sólo si f = k/N es racional)
"""

from fractions import Fraction

import numpy as np

__all__ = [
    "periodo_analogico",
    "velocidad_angular",
    "periodo_de_muestreo",
    "frecuencia_digital",
    "frecuencia_angular_digital",
    "frecuencia_analogica",
    "tiempo_de_muestra",
    "indice_de_muestra",
    "periodo_fundamental_digital",
    "frecuencia_alias",
    "resumen_senoidal",
]


# ---------------------------------------------------------------------------
# 1) Magnitudes de la señal ANALÓGICA
# ---------------------------------------------------------------------------
def periodo_analogico(F):
    """
    Periodo T (en segundos) de una señal analógica de frecuencia F (en Hz).

        T = 1 / F

    Ejemplo
    -------
    >>> periodo_analogico(50)      # la red eléctrica: 50 Hz
    0.02                           # 20 ms
    """
    T = 1 / F
    return T


def velocidad_angular(F):
    """
    Velocidad angular Ω (en rad/s) de una señal de frecuencia F (en Hz).

        Ω = 2π F

    Ejemplo
    -------
    >>> velocidad_angular(60)      # red eléctrica de 60 Hz
    376.99...                      # rad/s
    """
    Omega = 2 * np.pi * F
    return Omega


# ---------------------------------------------------------------------------
# 2) Magnitudes del MUESTREO
# ---------------------------------------------------------------------------
def periodo_de_muestreo(fs):
    """
    Periodo de muestreo Ts (en segundos): el tiempo que pasa entre una
    muestra y la siguiente.

        Ts = 1 / fs

    Ejemplo
    -------
    >>> periodo_de_muestreo(8000)
    0.000125                       # 125 microsegundos entre muestras
    """
    Ts = 1 / fs
    return Ts


def tiempo_de_muestra(n, fs):
    """
    ¿En qué instante (segundos) se tomó la muestra número n?

        t_n = n · Ts = n / fs

    'n' puede ser un número o un arreglo de números.

    Ejemplo
    -------
    >>> tiempo_de_muestra(4000, fs=8000)
    0.5                            # la muestra 4000 se tomó a los 0.5 s
    """
    Ts = periodo_de_muestreo(fs)
    t = n * Ts
    return t


def indice_de_muestra(t, fs):
    """
    Operación inversa: ¿qué número de muestra corresponde al instante t?

        n = t · fs     (redondeado al entero más cercano)

    Se redondea porque los índices SIEMPRE son números enteros.

    Ejemplo
    -------
    >>> indice_de_muestra(0.5, fs=8000)
    4000
    """
    n_real = t * fs                      # puede salir con decimales
    n_entero = np.round(n_real)          # el índice tiene que ser entero
    n_entero = np.asarray(n_entero).astype(int)
    if n_entero.ndim == 0:               # si nos dieron un solo número,
        n_entero = int(n_entero)         # regresamos un solo número
    return n_entero


# ---------------------------------------------------------------------------
# 3) Magnitudes de la señal DIGITAL
# ---------------------------------------------------------------------------
def frecuencia_digital(F, fs):
    """
    Frecuencia digital f (ciclos/muestra) de una senoidal de F Hz
    muestreada a fs muestras/segundo.

        f = F / fs

    Interpretación: "cuántos ciclos (o qué fracción de ciclo) avanza la
    señal entre una muestra y la siguiente".

    Ejemplo
    -------
    >>> frecuencia_digital(1000, 8000)
    0.125                          # 1/8 de ciclo por muestra -> 8 muestras por ciclo
    """
    f = F / fs
    return f


def frecuencia_angular_digital(F, fs):
    """
    Frecuencia angular digital ω (rad/muestra).

        ω = 2π f = 2π F / fs = Ω · Ts

    Ejemplo
    -------
    >>> frecuencia_angular_digital(1000, 8000)
    0.785...                       # π/4 rad por muestra
    """
    f = frecuencia_digital(F, fs)
    w = 2 * np.pi * f
    return w


def frecuencia_analogica(f, fs):
    """
    Operación inversa: dada la frecuencia digital f (ciclos/muestra) y fs,
    ¿qué frecuencia en Hz representa?

        F = f · fs
    """
    F = f * fs
    return F


def periodo_fundamental_digital(f, max_denominador=10000):
    """
    Periodo fundamental N (en muestras) de x[n] = cos(2π f n).

    Una senoidal DIGITAL sólo es periódica si f es un número RACIONAL:

        f = k / N     (fracción irreducible)   ->   periodo = N muestras

    Pasos:
      1. Escribimos f como fracción irreducible k/N.
      2. Si el denominador es "razonable", el periodo es N.
      3. Si no encontramos una fracción exacta, la señal NO es periódica
         (en la práctica: su periodo sería enorme) y regresamos None.

    Ejemplo
    -------
    >>> periodo_fundamental_digital(0.125)     # f = 1/8
    8
    >>> periodo_fundamental_digital(0.3)       # f = 3/10  -> 3 ciclos cada 10 muestras
    10
    >>> periodo_fundamental_digital(1/np.pi)   # f irracional -> no periódica
    None
    """
    # Paso 1: aproximar f por una fracción k/N con denominador limitado
    fraccion = Fraction(f).limit_denominator(max_denominador)

    # Paso 2: comprobar que la fracción representa EXACTAMENTE a f
    error = abs(float(fraccion) - f)
    if error > 1e-12:
        return None          # f no es (prácticamente) racional -> no periódica

    # Paso 3: el denominador de la fracción irreducible es el periodo
    N = fraccion.denominator
    return N


def frecuencia_alias(F, fs):
    """
    Frecuencia (en Hz) que "parece" tener una senoidal de F Hz después de
    muestrearla a fs. Siempre queda en el intervalo [0, fs/2].

    Si F <= fs/2 (se cumple Nyquist)  -> la frecuencia aparente es F.
    Si F >  fs/2 (NO se cumple)       -> aparece otra frecuencia: ALIAS.

    Pasos:
      1. Las frecuencias F y F + k·fs dan exactamente las mismas muestras,
         así que "doblamos" F al intervalo [0, fs):  F_r = F mod fs
      2. Las frecuencias F_r y fs - F_r también dan las mismas muestras
         (cos es una función par), así que si F_r > fs/2 usamos fs - F_r.

    Ejemplo
    -------
    >>> frecuencia_alias(7000, fs=8000)
    1000.0                       # una senoidal de 7 kHz suena como una de 1 kHz
    """
    # Paso 1: llevar F al intervalo [0, fs)
    F_reducida = F % fs

    # Paso 2: reflejar al intervalo [0, fs/2]
    if F_reducida > fs / 2:
        F_aparente = fs - F_reducida
    else:
        F_aparente = F_reducida

    return float(F_aparente)


def resumen_senoidal(F, fs):
    """
    Imprime una tabla con TODAS las magnitudes de una senoidal de F Hz
    muestreada a fs. Útil para repasar los conceptos.
    """
    T = periodo_analogico(F)
    Omega = velocidad_angular(F)
    Ts = periodo_de_muestreo(fs)
    f = frecuencia_digital(F, fs)
    w = frecuencia_angular_digital(F, fs)
    N = periodo_fundamental_digital(f)
    F_ap = frecuencia_alias(F, fs)

    print("=" * 62)
    print(f"  Senoidal de F = {F} Hz muestreada a fs = {fs} Hz")
    print("=" * 62)
    print("  SEÑAL ANALÓGICA")
    print(f"    Periodo              T  = 1/F      = {T:.6g} s")
    print(f"    Velocidad angular    Ω  = 2πF      = {Omega:.6g} rad/s")
    print("  MUESTREO")
    print(f"    Periodo de muestreo  Ts = 1/fs     = {Ts:.6g} s")
    print(f"    Muestras por ciclo   T/Ts = fs/F   = {fs / F:.6g}")
    print("  SEÑAL DIGITAL")
    print(f"    Frecuencia digital   f  = F/fs     = {f:.6g} ciclos/muestra")
    print(f"    Frec. angular dig.   ω  = 2πf      = {w:.6g} rad/muestra  (= {w / np.pi:.4g}·π)")
    if N is None:
        print("    Periodo fundamental  N  = NO es periódica (f no es racional)")
    else:
        print(f"    Periodo fundamental  N            = {N} muestras")
    print("  NYQUIST")
    if F <= fs / 2:
        print(f"    F = {F} Hz <= fs/2 = {fs / 2} Hz  ->  se cumple Nyquist")
    else:
        print(f"    F = {F} Hz >  fs/2 = {fs / 2} Hz  ->  hay aliasing: se verá como {F_ap} Hz")
    print("=" * 62)
