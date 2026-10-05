"""
frecuencia.py
=============
Análisis en frecuencia: DFT, FFT y espectro.

DFT (Transformada Discreta de Fourier) de N puntos:

    X[k] = Σ_{n=0}^{N-1} x[n] · e^{-j 2π k n / N},     k = 0, 1, ..., N-1

    * X[k] es un número COMPLEJO: |X[k]| = magnitud, ∠X[k] = fase.
    * El índice k corresponde a la frecuencia  F_k = k · fs / N  (Hz).
    * La resolución en frecuencia es Δf = fs / N: para distinguir
      frecuencias cercanas necesitamos MÁS muestras (más tiempo de señal).
    * Para señales reales, la mitad superior (k > N/2) es un "espejo" de
      la mitad inferior: corresponde a las frecuencias negativas.

La FFT es un ALGORITMO RÁPIDO para calcular exactamente la misma DFT
(N·log2(N) operaciones en lugar de N²).
"""

import numpy as np

__all__ = [
    "dft",
    "eje_frecuencias",
    "espectro",
    "espectro_2d",
]


def dft(x):
    """
    DFT calculada DIRECTAMENTE con la fórmula (dos ciclos for).
    Es lenta (N² operaciones), pero sirve para entender qué hace la FFT.

    Pasos:
      Para cada frecuencia k:
        comparamos la señal con una "senoidal compleja" e^{-j2πkn/N}
        multiplicando muestra a muestra y sumando.
    """
    x = np.asarray(x)
    N = len(x)
    X = np.zeros(N, dtype=complex)

    for k in range(N):                         # para cada frecuencia k
        suma = 0.0 + 0.0j
        for n in range(N):                     # recorremos la señal
            exponente = -2j * np.pi * k * n / N
            suma = suma + x[n] * np.exp(exponente)
        X[k] = suma

    return X


def eje_frecuencias(N, fs):
    """
    Frecuencia en Hz de cada índice k de la DFT:

        F_k = k · fs / N,     k = 0, 1, ..., N-1
    """
    k = np.arange(0, N)
    F = k * fs / N
    return F


def espectro(x, fs):
    """
    Espectro de magnitud de UN LADO (sólo frecuencias de 0 a fs/2), escalado
    para que una senoidal de amplitud A aparezca con altura A.

    Pasos:
      1. Calcular la FFT.
      2. Magnitud |X[k]| y dividir entre N.
      3. Quedarnos con la mitad: k = 0 ... N/2  (0 Hz ... fs/2).
      4. Multiplicar por 2 (excepto en 0 Hz y en fs/2), porque la energía de
         cada senoidal real se reparte entre +F y -F, y tiramos la mitad negativa.

    Regresa
    -------
    frecuencias : arreglo de frecuencias en Hz (0 ... fs/2)
    magnitud    : amplitud de cada componente
    """
    x = np.asarray(x, dtype=float)
    N = len(x)

    # Paso 1: FFT
    X = np.fft.fft(x)

    # Paso 2: magnitud normalizada
    magnitud_completa = np.abs(X) / N

    # Paso 3: sólo la mitad positiva
    mitad = N // 2 + 1
    magnitud = magnitud_completa[0:mitad].copy()
    frecuencias = eje_frecuencias(N, fs)[0:mitad]

    # Paso 4: duplicar todas menos 0 Hz (y fs/2 si N es par)
    magnitud[1:] = 2 * magnitud[1:]
    if N % 2 == 0:
        magnitud[-1] = magnitud[-1] / 2

    return frecuencias, magnitud


def espectro_2d(imagen):
    """
    Espectro de magnitud de una imagen (FFT en 2D), centrado y en escala
    logarítmica para poder verlo.

    Pasos:
      1. FFT 2D: aplica la FFT a cada fila y luego a cada columna.
      2. fftshift: mueve la frecuencia (0,0) al CENTRO de la imagen.
         - Centro            -> frecuencias bajas (zonas suaves, el "promedio")
         - Lejos del centro  -> frecuencias altas (bordes, detalles, ruido)
      3. Magnitud y logaritmo: log(1 + |F|), porque el valor del centro es
         muchísimo más grande que el resto.

    Regresa
    -------
    F_centrada : la FFT 2D compleja ya centrada (para filtrar)
    magnitud_log : log(1 + |F|) para graficar
    """
    # Paso 1
    F = np.fft.fft2(imagen)
    # Paso 2
    F_centrada = np.fft.fftshift(F)
    # Paso 3
    magnitud_log = np.log(1 + np.abs(F_centrada))
    return F_centrada, magnitud_log