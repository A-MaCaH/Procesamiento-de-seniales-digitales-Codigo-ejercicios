"""
correlacion.py
==============
Correlación: mide qué tanto se PARECEN dos señales cuando desplazamos una
de ellas l muestras ("lag" o retardo).

    Correlación cruzada:   r_xy[l] = Σ_n  x[n] · y[n - l]
    Autocorrelación:       r_xx[l] = Σ_n  x[n] · x[n - l]

Comparación con la convolución:
    convolución   y[n] = Σ x[k] · h[n - k]   -> se INVIERTE h
    correlación   r[l] = Σ x[n] · y[n - l]   -> NO se invierte

    De hecho:  r_xy[l] = x[l] * y[-l]   (convolución con y invertida)

Aplicaciones: encontrar retardos (radar, sonar, GPS), detectar un patrón
dentro de una señal, encontrar el periodo de una señal con ruido.
"""

import numpy as np

__all__ = [
    "correlacion_cruzada",
    "autocorrelacion",
    "correlacion_normalizada",
    "estimar_retardo",
]


def correlacion_cruzada(x, y):
    """
    Correlación cruzada  r_xy[l] = Σ_n x[n] · y[n - l]

    Suponemos que x y y empiezan en n = 0.

    Regresa
    -------
    lags : los desplazamientos l, desde -(len(y)-1) hasta len(x)-1
    r    : el valor de la correlación para cada l

    Pasos:
      Para cada desplazamiento l:
        1. Buscamos las n en las que AMBAS muestras existen:
               0 <= n < Lx     y     0 <= n - l < Ly
           es decir   max(0, l) <= n < min(Lx, Ly + l)
        2. Multiplicamos esos pedazos de x y de y muestra a muestra y sumamos.

    (Es exactamente el doble ciclo del notebook 07, pero el ciclo interno
    se hace con rebanadas de numpy para que funcione rápido con señales
    largas como audio o ECG.)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    Lx = len(x)
    Ly = len(y)

    # Todos los desplazamientos posibles en los que las señales se "tocan"
    lags = np.arange(-(Ly - 1), Lx)
    r = np.zeros(len(lags))

    for i in range(len(lags)):
        l = lags[i]

        # Paso 1: rango de n donde x[n] y y[n - l] existen
        n_ini = max(0, l)
        n_fin = min(Lx, Ly + l)

        # Paso 2: pedazo de x (x[n]) y pedazo de y (y[n - l]) que se enciman
        pedazo_x = x[n_ini:n_fin]
        pedazo_y = y[n_ini - l:n_fin - l]
        r[i] = np.sum(pedazo_x * pedazo_y)

    return lags, r


def autocorrelacion(x):
    """
    Autocorrelación  r_xx[l] = Σ_n x[n] · x[n - l]

    Propiedades que conviene comprobar:
      * Es simétrica:          r_xx[-l] = r_xx[l]
      * Su máximo está en l=0  y vale la ENERGÍA de x.
      * Si x es periódica con periodo N, tiene picos en l = N, 2N, ...
    """
    lags, r = correlacion_cruzada(x, x)
    return lags, r


def correlacion_normalizada(x, y):
    """
    Correlación cruzada dividida entre √(Ex · Ey) para que quede entre -1 y 1:

        ρ[l] = r_xy[l] / √(Ex · Ey)

        ρ =  1 -> las señales son iguales (salvo una escala positiva)
        ρ = -1 -> son iguales pero con signo contrario
        ρ ≈  0 -> no se parecen
    """
    lags, r = correlacion_cruzada(x, y)
    Ex = np.sum(np.asarray(x, dtype=float) ** 2)
    Ey = np.sum(np.asarray(y, dtype=float) ** 2)
    rho = r / np.sqrt(Ex * Ey)
    return lags, rho


def estimar_retardo(x_original, y_recibida):
    """
    Si y_recibida es una copia retrasada (y tal vez ruidosa) de x_original,

        y[n] ≈ a · x[n - D]

    la correlación r_yx[l] = Σ y[n]·x[n-l] es MÁXIMA en l = D.

    Regresa
    -------
    D : retardo estimado en muestras.
    """
    lags, r = correlacion_cruzada(y_recibida, x_original)
    posicion_del_maximo = np.argmax(r)
    D = int(lags[posicion_del_maximo])
    return D
