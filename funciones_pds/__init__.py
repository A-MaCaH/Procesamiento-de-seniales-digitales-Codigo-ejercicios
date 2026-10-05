"""
funciones_pds
=============
Biblioteca de funciones para la clase de Procesamiento Digital de Señales (PDS).

Cada archivo agrupa las funciones de un tema:

    conceptos.py        -> frecuencia, periodo, frecuencia digital, tiempo de una muestra...
    senales_basicas.py  -> impulso, escalón, rampa, exponencial, senoidales
    archivos.py         -> cargar/guardar audio, imágenes y señales biomédicas; reproducir audio
    graficas.py         -> funciones para dibujar señales, espectros e imágenes
    propiedades.py      -> energía, potencia, periodo de una señal
    operaciones.py      -> suma, producto, desplazamiento, inversión, escalamiento en el tiempo
    sistemas.py         -> ecuación en diferencias, respuesta al impulso, propiedades de sistemas
    convolucion.py      -> convolución en 1D y en 2D, kernels para imágenes
    correlacion.py      -> correlación cruzada, autocorrelación, estimación de retardos
    frecuencia.py       -> DFT, FFT, espectro de una señal, FFT en 2D
    verificacion.py     -> funciones para que comprueben sus resultados en los ejercicios

Cómo usarla desde un notebook que está en la carpeta "notebooks" o "ejercicios":

    import sys
    sys.path.append("..")          # le decimos a Python que busque una carpeta "arriba"
    import funciones_pds as pds    # importamos la biblioteca con un nombre corto

    n, x = pds.impulso(-5, 5)      # y usamos cualquier función con "pds."
"""

from .conceptos import *
from .senales_basicas import *
from .archivos import *
from .graficas import *
from .propiedades import *
from .operaciones import *
from .sistemas import *
from .convolucion import *
from .correlacion import *
from .frecuencia import *
from .verificacion import *
