"""
verificacion.py
===============
Funciones para que el alumno COMPRUEBE sus resultados en los ejercicios.
En lugar de un error difícil de leer, imprimen un mensaje claro.
"""

import numpy as np

__all__ = ["comprobar", "comprobar_iguales"]


def comprobar(condicion, descripcion):
    """
    Imprime "Correcto" si la condición es verdadera, o "Revisa" si es falsa.

    Ejemplo
    -------
    >>> comprobar(N == 8, "El periodo debe ser 8 muestras")
    """
    if condicion:
        print(f"Correcto: {descripcion}")
    else:
        print(f"Revisa:   {descripcion}")
    return bool(condicion)


def comprobar_iguales(obtenido, esperado, descripcion, tolerancia=1e-6):
    """
    Compara dos números o dos arreglos (con una tolerancia, porque las
    computadoras redondean) e imprime el resultado.
    """
    obtenido = np.asarray(obtenido, dtype=complex)
    esperado = np.asarray(esperado, dtype=complex)

    if obtenido.shape != esperado.shape:
        print(f"Revisa:   {descripcion}")
        print(f"   Los tamaños no coinciden: obtuviste {obtenido.shape}, se esperaba {esperado.shape}")
        return False

    diferencia = np.max(np.abs(obtenido - esperado)) if obtenido.size > 0 else 0.0
    if diferencia < tolerancia:
        print(f"Correcto: {descripcion}")
        return True

    print(f"Revisa:   {descripcion}")
    print(f"   Diferencia máxima: {diferencia:.4g}")
    return False
