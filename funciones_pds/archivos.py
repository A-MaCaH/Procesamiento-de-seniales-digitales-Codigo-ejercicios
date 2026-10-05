"""
archivos.py
===========
Funciones para CARGAR, GUARDAR y REPRODUCIR señales reales:

    * Audio ................ archivos .wav
    * Imágenes ............. archivos .png / .jpg
    * Señales biomédicas ... archivos de texto .csv / .txt (ECG, EMG, EEG, ...)

Todas las funciones regresan los datos como arreglos de numpy con
números decimales (float), para que podamos operar con ellos.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

__all__ = [
    "cargar_audio",
    "guardar_audio",
    "reproducir",
    "cargar_imagen",
    "cargar_senal_texto",
    "guardar_senal_texto",
]


# ---------------------------------------------------------------------------
# AUDIO
# ---------------------------------------------------------------------------
def cargar_audio(ruta):
    """
    Carga un archivo de audio .wav

    Regresa
    -------
    fs : int     Frecuencia de muestreo (muestras por segundo).
    x  : arreglo Señal de audio en MONO con valores entre -1 y 1.

    Pasos:
      1. Leer el archivo (scipy nos da fs y las muestras).
      2. Convertir las muestras enteras a decimales entre -1 y 1.
      3. Si el audio es estéreo (2 columnas: izquierda y derecha),
         promediamos ambos canales para tener una sola señal.

    Ejemplo
    -------
    >>> fs, x = cargar_audio("../datos/nota_la440.wav")
    """
    # Paso 1: leer el archivo
    fs, datos = wavfile.read(ruta)

    # Paso 2: pasar a decimales entre -1 y 1 según el tipo de dato
    if datos.dtype == np.int16:            # el formato más común (16 bits)
        x = datos / 32768.0
    elif datos.dtype == np.int32:          # 32 bits
        x = datos / 2147483648.0
    elif datos.dtype == np.uint8:          # 8 bits (va de 0 a 255, centrado en 128)
        x = (datos - 128.0) / 128.0
    else:                                  # ya viene en decimales
        x = datos.astype(float)

    # Paso 3: estéreo -> mono
    if x.ndim == 2:
        print(f"El audio tiene {x.shape[1]} canales; se promedian para obtener una señal mono.")
        x = np.mean(x, axis=1)

    return fs, x


def guardar_audio(ruta, x, fs):
    """
    Guarda la señal x como archivo .wav de 16 bits.

    Pasos:
      1. Normalizar: dividir entre el valor absoluto máximo para que la
         señal quede entre -1 y 1 (si no, el sonido se "satura").
      2. Convertir a enteros de 16 bits (de -32767 a 32767).
      3. Escribir el archivo.
    """
    x = np.asarray(x, dtype=float)

    # Paso 1: normalizar
    maximo = np.max(np.abs(x))
    if maximo > 0:
        x = x / maximo

    # Paso 2: a enteros de 16 bits
    x_entero = np.int16(x * 32767)

    # Paso 3: escribir
    wavfile.write(ruta, int(fs), x_entero)
    print(f"Audio guardado en: {ruta}  ({len(x)} muestras, fs = {fs} Hz)")


def reproducir(x, fs):
    """
    Crea un reproductor de audio dentro del notebook de Jupyter.

    IMPORTANTE: debe ser la ÚLTIMA línea de la celda, o usar display():
        reproducir(x, fs)
        display(reproducir(x, fs))
    """
    from IPython.display import Audio   # sólo existe dentro de Jupyter/IPython
    return Audio(data=np.asarray(x, dtype=float), rate=int(fs))


# ---------------------------------------------------------------------------
# IMÁGENES
# ---------------------------------------------------------------------------
def cargar_imagen(ruta, en_grises=True):
    """
    Carga una imagen como una MATRIZ de números (una señal en 2D).

        imagen[fila, columna]  -> intensidad del pixel, entre 0 (negro) y 1 (blanco)

    Parámetros
    ----------
    ruta      : str   Archivo .png o .jpg
    en_grises : bool  Si es True, convierte la imagen a escala de grises
                      (una sola matriz). Si es False, deja los 3 canales R, G, B.

    Pasos:
      1. Leer la imagen.
      2. Pasar los valores al rango 0..1 (los .jpg vienen de 0..255).
      3. Si se pide, convertir a grises con la fórmula de luminancia:
            gris = 0.299·R + 0.587·G + 0.114·B
         (el ojo humano es más sensible al verde).
    """
    # Paso 1: leer
    imagen = plt.imread(ruta)

    # Paso 2: rango 0..1
    if imagen.dtype == np.uint8:
        imagen = imagen / 255.0
    else:
        imagen = imagen.astype(float)

    # Paso 3: a escala de grises (si la imagen es a color)
    if en_grises and imagen.ndim == 3:
        R = imagen[:, :, 0]
        G = imagen[:, :, 1]
        B = imagen[:, :, 2]
        imagen = 0.299 * R + 0.587 * G + 0.114 * B

    filas = imagen.shape[0]
    columnas = imagen.shape[1]
    print(f"Imagen cargada: {filas} filas x {columnas} columnas")
    return imagen


# ---------------------------------------------------------------------------
# SEÑALES BIOMÉDICAS (o cualquier señal guardada en texto)
# ---------------------------------------------------------------------------
def cargar_senal_texto(ruta, fs, columna=0, delimitador=",", saltar_filas=1):
    """
    Carga una señal guardada en un archivo de texto (.csv o .txt),
    como las que se descargan de PhysioNet o de un osciloscopio.

    Parámetros
    ----------
    ruta         : str    Archivo a leer.
    fs           : float  Frecuencia de muestreo con la que se adquirió la señal.
                          (normalmente el archivo no la trae; hay que buscarla en la documentación)
    columna      : int    Qué columna del archivo contiene la señal (0 = la primera).
    delimitador  : str    Carácter que separa las columnas ("," o ";" o "\\t").
    saltar_filas : int    Cuántas filas de encabezado ignorar.

    Regresa
    -------
    t : tiempo de cada muestra en segundos (t = n / fs)
    x : valores de la señal
    """
    # Paso 1: leer la tabla de números
    datos = np.loadtxt(ruta, delimiter=delimitador, skiprows=saltar_filas)

    # Paso 2: quedarnos con la columna que nos interesa
    if datos.ndim == 1:
        x = datos
    else:
        x = datos[:, columna]

    # Paso 3: construir el eje de tiempo
    n = np.arange(0, len(x))
    t = n / fs

    print(f"Señal cargada: {len(x)} muestras, duración = {len(x) / fs:.2f} s")
    return t, x


def guardar_senal_texto(ruta, x, encabezado="valor"):
    """
    Guarda una señal en un archivo de texto con una columna.
    """
    np.savetxt(ruta, np.asarray(x), delimiter=",", header=encabezado, comments="")
    print(f"Señal guardada en: {ruta}")
