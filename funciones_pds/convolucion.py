"""
convolucion.py
==============
Convolución en una dimensión (señales) y en dos dimensiones (imágenes).

Convolución 1D
--------------
    y[n] = x[n] * h[n] = Σ_k  x[k] · h[n - k]

    Para un sistema lineal e invariante en el tiempo, h[n] es la respuesta
    al impulso y y[n] es la salida para la entrada x[n].

Convolución 2D
--------------
    y[m, n] = x[m, n] * h[m, n] = Σ_i Σ_j  h[i, j] · x[m - i, n - j]

    Es la misma idea con dos índices: la imagen x se combina con una matriz
    pequeña h, a la que llamamos "kernel" o "máscara". Según el kernel, el
    resultado suaviza la imagen, detecta bordes o la hace más nítida.

Dos formas de ver la misma operación
------------------------------------
    1. "Invertir y deslizar": para cada muestra de salida, invertimos h, la
       colocamos sobre x, multiplicamos punto a punto y sumamos.
    2. "Suma de copias": cada muestra de h produce una copia de x desplazada y
       escalada; la salida es la suma de todas esas copias.

    convolucion() usa la forma 1 (es la fórmula que se escribe en clase).
    convolucion_2d() usa la forma 2, porque con imágenes la forma 1 requiere
    cuatro ciclos anidados y tarda mucho.
"""

import numpy as np

__all__ = [
    "convolucion",
    "convolucion_2d",
    "kernel_promedio",
    "kernel_gaussiano",
    "kernel_sobel",
    "kernel_laplaciano",
]


# ---------------------------------------------------------------------------
# Convolución 1D
# ---------------------------------------------------------------------------
def convolucion(n_x, x, n_h, h):
    """
    Convolución discreta en 1D, implementada con la fórmula:

        y[n] = Σ_k  x[k] · h[n - k]

    Respeta los índices:
      * el primer índice de y es  n_x[0] + n_h[0]
      * la longitud de y es       len(x) + len(h) - 1

    Pasos:
      1. Calculamos la longitud y el eje de índices de la salida.
      2. Para cada muestra de salida m:
           recorremos todas las muestras de x (posición k del arreglo)
           y, si h[m - k] existe, acumulamos x[k]·h[m - k].

    Para señales largas (por ejemplo, audio) esta función es lenta porque
    hace len(x)·len(h) multiplicaciones con ciclos de Python. En ese caso se
    usa np.convolve(x, h), que da los mismos valores (pero no calcula índices).

    Regresa
    -------
    n_y, y

    Ejemplo
    -------
    >>> n_y, y = convolucion([0, 1, 2], [1, 2, 1], [0, 1], [1, 1])
    >>> y
    array([1., 3., 3., 1.])
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    Lx = len(x)
    Lh = len(h)

    # Paso 1: longitud y eje de la salida
    Ly = Lx + Lh - 1
    n_inicio = n_x[0] + n_h[0]
    n_y = np.arange(n_inicio, n_inicio + Ly)
    y = np.zeros(Ly)

    # Paso 2: la suma de convolución (trabajamos con posiciones del arreglo 0, 1, 2, ...)
    for m in range(Ly):                 # para cada muestra de salida
        suma = 0.0
        for k in range(Lx):             # recorremos la entrada
            j = m - k                   # posición de h que coincide con x[k]
            if 0 <= j < Lh:             # sólo si esa posición existe
                suma = suma + x[k] * h[j]
        y[m] = suma

    return n_y, y


# ---------------------------------------------------------------------------
# Convolución 2D
# ---------------------------------------------------------------------------
def convolucion_2d(imagen, kernel, modo="igual"):
    """
    Convolución en 2D de una imagen con un kernel:

        y[m, n] = Σ_i Σ_j  kernel[i, j] · imagen[m - i, n - j]

    Parámetros
    ----------
    imagen : matriz de M x N
    kernel : matriz pequeña de P x Q (por ejemplo 3 x 3)
    modo   : "completa" -> la salida mide (M + P - 1) x (N + Q - 1), igual que en 1D.
             "igual"    -> la salida mide M x N (se recorta el centro de la completa),
                           que es lo que normalmente queremos con imágenes.

    Pasos (forma "suma de copias"):
      1. Creamos una salida de ceros del tamaño completo.
      2. Para cada elemento kernel[i, j]:
           sumamos a la salida una copia de la imagen, desplazada i filas y
           j columnas, y multiplicada por kernel[i, j].
      3. Si el modo es "igual", recortamos la parte central.

    Es exactamente la fórmula de arriba, sólo que en lugar de recorrer pixel
    por pixel recorremos los elementos del kernel (que son pocos).

    Regresa
    -------
    y : matriz con el resultado.
    """
    imagen = np.asarray(imagen, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    M, N = imagen.shape
    P, Q = kernel.shape

    # Paso 1: salida completa llena de ceros
    y = np.zeros((M + P - 1, N + Q - 1))

    # Paso 2: sumar una copia desplazada y escalada de la imagen por cada elemento del kernel
    for i in range(P):
        for j in range(Q):
            y[i:i + M, j:j + N] = y[i:i + M, j:j + N] + kernel[i, j] * imagen

    # Paso 3: recortar el centro si se pide el mismo tamaño que la imagen
    if modo == "igual":
        fila_inicio = (P - 1) // 2
        columna_inicio = (Q - 1) // 2
        y = y[fila_inicio:fila_inicio + M, columna_inicio:columna_inicio + N]
    elif modo != "completa":
        raise ValueError('El modo debe ser "igual" o "completa".')

    return y


# ---------------------------------------------------------------------------
# Kernels más usados
# ---------------------------------------------------------------------------
def kernel_promedio(tamano):
    """
    Kernel de promedio de tamano x tamano: todos sus elementos valen 1/tamano².
    Reemplaza cada pixel por el promedio de sus vecinos: suaviza la imagen
    (es un filtro pasa-bajas). Sus elementos suman 1, así que no cambia el
    brillo promedio de la imagen.
    """
    kernel = np.ones((tamano, tamano)) / (tamano * tamano)
    return kernel


def kernel_gaussiano(tamano, sigma):
    """
    Kernel gaussiano de tamano x tamano (tamano impar):

        g[i, j] = exp( -(i² + j²) / (2·sigma²) ),   i, j medidos desde el centro

    También suaviza, pero da más peso a los vecinos cercanos, por lo que el
    resultado se ve más natural que con el promedio. Se normaliza para que
    sus elementos sumen 1.
    """
    centro = tamano // 2
    kernel = np.zeros((tamano, tamano))
    for i in range(tamano):
        for j in range(tamano):
            distancia2 = (i - centro) ** 2 + (j - centro) ** 2
            kernel[i, j] = np.exp(-distancia2 / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)
    return kernel


def kernel_sobel(direccion="horizontal"):
    """
    Kernel de Sobel para detectar bordes (aproxima una derivada).

    direccion = "horizontal": responde a cambios de intensidad de izquierda a
                              derecha, es decir, detecta bordes verticales.
    direccion = "vertical":   responde a cambios de arriba hacia abajo, es
                              decir, detecta bordes horizontales.

    Sus elementos suman 0: en una zona de intensidad constante la salida es 0.
    """
    sobel = np.array([[1.0, 0.0, -1.0],
                      [2.0, 0.0, -2.0],
                      [1.0, 0.0, -1.0]])
    if direccion == "horizontal":
        return sobel
    if direccion == "vertical":
        return sobel.T                  # .T es la transpuesta: filas por columnas
    raise ValueError('La dirección debe ser "horizontal" o "vertical".')


def kernel_laplaciano():
    """
    Kernel laplaciano: aproxima la segunda derivada en las dos direcciones.
    Resalta los bordes en todas las direcciones. Sus elementos suman 0.

        [ 0   1   0 ]
        [ 1  -4   1 ]
        [ 0   1   0 ]
    """
    kernel = np.array([[0.0, 1.0, 0.0],
                       [1.0, -4.0, 1.0],
                       [0.0, 1.0, 0.0]])
    return kernel
