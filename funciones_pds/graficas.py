"""
graficas.py
===========
Funciones para DIBUJAR señales. Regla del curso:

    * Señales DISCRETAS  x[n]  -> se dibujan con  stem  (palitos con bolita).
    * Señales CONTINUAS  x(t)  -> se dibujan con  plot (línea continua).

Así nunca confundimos una señal digital con una analógica.
"""

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "graficar_discreta",
    "graficar_continua",
    "graficar_muestreo",
    "graficar_espectro",
    "mostrar_imagen",
]


def graficar_discreta(n, x, titulo="Señal discreta", etiqueta_x="n (muestras)",
                      etiqueta_y="x[n]", ax=None, color="C0"):
    """
    Dibuja una señal discreta x[n] con stem.

    Parámetros
    ----------
    n, x  : índices y valores de la señal.
    titulo, etiqueta_x, etiqueta_y : textos de la gráfica.
    ax    : (opcional) ejes de matplotlib donde dibujar, para hacer subgráficas.
    """
    # Si no nos dan unos ejes, creamos una figura nueva
    if ax is None:
        figura, ax = plt.subplots(figsize=(8, 3))

    ax.stem(n, x, linefmt=color + "-", markerfmt=color + "o", basefmt="k-")
    ax.set_title(titulo)
    ax.set_xlabel(etiqueta_x)
    ax.set_ylabel(etiqueta_y)
    ax.grid(True, alpha=0.3)
    return ax


def graficar_continua(t, x, titulo="Señal", etiqueta_x="t (s)",
                      etiqueta_y="x(t)", ax=None, color="C0"):
    """
    Dibuja una señal con línea continua (para señales analógicas o para
    señales digitales con MUCHAS muestras, como audio o ECG).
    """
    if ax is None:
        figura, ax = plt.subplots(figsize=(8, 3))

    ax.plot(t, x, color=color)
    ax.set_title(titulo)
    ax.set_xlabel(etiqueta_x)
    ax.set_ylabel(etiqueta_y)
    ax.grid(True, alpha=0.3)
    return ax


def graficar_muestreo(t_continua, x_continua, t_muestras, x_muestras,
                      titulo="Muestreo", ax=None):
    """
    Dibuja la señal analógica (línea) y encima sus muestras (stem).
    Ideal para VER el proceso de muestreo.
    """
    if ax is None:
        figura, ax = plt.subplots(figsize=(8, 3))

    ax.plot(t_continua, x_continua, color="0.6", label="x(t) analógica")
    ax.stem(t_muestras, x_muestras, linefmt="C3-", markerfmt="C3o", basefmt="k-",
            label="x[n] = x(n·Ts) muestras")
    ax.set_title(titulo)
    ax.set_xlabel("t (s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    return ax


def graficar_espectro(frecuencias, magnitud, titulo="Espectro de magnitud",
                      etiqueta_x="Frecuencia (Hz)", ax=None, discreto=False):
    """
    Dibuja la magnitud del espectro.

    discreto=True dibuja con stem (útil cuando hay pocos puntos de la DFT).
    """
    if ax is None:
        figura, ax = plt.subplots(figsize=(8, 3))

    if discreto:
        ax.stem(frecuencias, magnitud, basefmt="k-")
    else:
        ax.plot(frecuencias, magnitud)

    ax.set_title(titulo)
    ax.set_xlabel(etiqueta_x)
    ax.set_ylabel("|X|")
    ax.grid(True, alpha=0.3)
    return ax


def mostrar_imagen(imagen, titulo="Imagen", ax=None, barra_color=False):
    """
    Muestra una imagen (matriz) en escala de grises.
    """
    if ax is None:
        figura, ax = plt.subplots(figsize=(5, 5))

    if imagen.ndim == 2:
        dibujo = ax.imshow(imagen, cmap="gray")
    else:
        dibujo = ax.imshow(np.clip(imagen, 0, 1))

    ax.set_title(titulo)
    ax.set_xlabel("columna")
    ax.set_ylabel("fila")
    if barra_color:
        plt.colorbar(dibujo, ax=ax)
    return ax
