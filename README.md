# Procesamiento Digital de Señales con Python

Este repositorio contiene el material de programación del curso de Procesamiento Digital de Señales.
No necesitas saber programar bien para usarlo. La idea es avanzar poco a poco: primero hacemos cada cálculo
"a mano", con una operación por línea; después lo guardamos en una función; y al final usamos esas funciones
para trabajar con señales reales, como audio, imágenes y electrocardiogramas.

## Cómo empezar

Vamos a trabajar en Google Colab, así que no tienes que instalar nada en tu computadora.

1. Abre el enlace del notebook que te indique tu profesor (están en la tabla de temas, más abajo).
2. Ejecuta la primera celda de código con `Shift + Enter`. Esa celda descarga lo que hace falta y, cuando termina,
   imprime `Bibliotecas cargadas. Ya puedes continuar.`
3. Ejecuta las celdas en orden, de arriba hacia abajo, y lee lo que imprime cada una.
4. Para guardar tu trabajo usa el menú Archivo > Guardar una copia en Drive. Si no lo haces, tus cambios se pierden
   al cerrar la pestaña.

Si algo empieza a fallar sin razón aparente, usa Entorno de ejecución > Reiniciar y ejecutar todo. Muchas veces el
problema es que alguna celda anterior no se ejecutó.

## Cómo estudiar cada tema

Cada tema tiene dos notebooks y conviene trabajarlos en este orden.

**1. Teoría** (carpeta `notebooks/`). Explica el tema y resuelve ejemplos paso a paso. Al final hay unas preguntas
para comprobar si entendiste; intenta contestarlas antes de abrir las respuestas.

**2. Ejercicios** (carpeta `ejercicios/`). Aquí te toca a ti:

- Busca las marcas `COMPLETA` y cambia los `...` por tu código. La marca suele venir con una pista.
- Después de cada ejercicio hay una celda de verificación. Si tu resultado es correcto imprime `Correcto`;
  si no, imprime `Revisa` y te dice qué parte no coincide.
- Contesta las preguntas de reflexión con tus propias palabras. Es la parte más importante: el código es la
  herramienta, pero lo que queremos es entender qué le pasa a la señal.

## Temas

| # | Tema | Teoría | Ejercicios |
|---|---|---|---|
| 00 | Python mínimo para PDS | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/notebooks/00_python_minimo.ipynb) | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/ejercicios/00_python_minimo_ejercicios.ipynb) |
| 01 | Senoidales: frecuencia, periodo, muestreo y Nyquist | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/notebooks/01_senoidales_y_frecuencia.ipynb) | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/ejercicios/01_senoidales_y_frecuencia_ejercicios.ipynb) |
| 02 | Señales básicas: impulso, escalón y suma de impulsos | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/notebooks/02_senales_basicas_e_impulsos.ipynb) | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/ejercicios/02_senales_basicas_e_impulsos_ejercicios.ipynb) |
| 03 | Señales reales: audio, imágenes y ECG | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/notebooks/03_cargar_ver_escuchar.ipynb) | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/ejercicios/03_cargar_ver_escuchar_ejercicios.ipynb) |
| 04 | Periodo, energía y potencia | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/notebooks/04_periodo_energia_potencia.ipynb) | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/ejercicios/04_periodo_energia_potencia_ejercicios.ipynb) |
| 05 | Operaciones entre señales | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/notebooks/05_operaciones_entre_senales.ipynb) | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/ejercicios/05_operaciones_entre_senales_ejercicios.ipynb) |
| 06 | Sistemas, respuesta al impulso y convolución | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/notebooks/06_sistemas_y_convolucion.ipynb) | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/ejercicios/06_sistemas_y_convolucion_ejercicios.ipynb) |
| 07 | Correlación | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/notebooks/07_correlacion.ipynb) | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/ejercicios/07_correlacion_ejercicios.ipynb) |
| 08 | La DFT y la FFT | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/notebooks/08_fft.ipynb) | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/ejercicios/08_fft_ejercicios.ipynb) |
| 09 | FFT en 2D: imágenes en frecuencia | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/notebooks/09_fft_2d.ipynb) | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/ejercicios/09_fft_2d_ejercicios.ipynb) |
| 10 | Convolución en 1D y 2D | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/notebooks/10_convolucion_1d_2d.ipynb) | [abrir](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/ejercicios/10_convolucion_1d_2d_ejercicios.ipynb) |

## La biblioteca del curso: `funciones_pds`

Las funciones que vamos construyendo en los notebooks están guardadas en la carpeta `funciones_pds/`, ordenadas
por tema. Las puedes usar en tus ejercicios o copiarlas a tus propios programas.

| Archivo | Qué contiene |
|---|---|
| `conceptos.py` | frecuencia, periodo, velocidad angular, frecuencia digital, tiempo de una muestra, alias |
| `senales_basicas.py` | impulso, escalón, rampa, exponencial, senoidales, suma de impulsos |
| `archivos.py` | cargar y guardar audio (.wav), imágenes y señales en texto (ECG); reproducir audio |
| `graficas.py` | gráficas de señales discretas y continuas, muestreo, espectros e imágenes |
| `propiedades.py` | energía, potencia, valor RMS y periodo |
| `operaciones.py` | sumar, multiplicar, desplazar, invertir, submuestrear, sobremuestrear, parte par e impar |
| `sistemas.py` | ecuación en diferencias, respuesta al impulso, linealidad, invarianza, estabilidad |
| `convolucion.py` | convolución en 1D y en 2D; kernels de promedio, gaussiano, Sobel y laplaciano |
| `correlacion.py` | correlación cruzada, autocorrelación y estimación de retardos |
| `frecuencia.py` | DFT, espectro con FFT y FFT en 2D |
| `verificacion.py` | las funciones `comprobar` y `comprobar_iguales` que usan las celdas de verificación |

Dentro de un notebook se usan con el prefijo `pds.`. Por ejemplo:

```python
n, x = pds.impulso(-5, 5)                    # crear un impulso entre n = -5 y n = 5
pds.graficar_discreta(n, x, titulo="δ[n]")   # graficarlo

help(pds.convolucion)                        # leer la explicación de una función

import inspect
print(inspect.getsource(pds.convolucion))    # ver su código completo
```

## Reglas que seguimos en todo el curso

1. Una señal discreta se guarda en dos arreglos: `n`, con los índices de tiempo, y `x`, con los valores.
   Recuerda que `x[0]` en Python es el primer valor guardado, no necesariamente el valor en el instante `n = 0`.
2. Las señales discretas se grafican con `stem` (líneas verticales con un punto) y las continuas con `plot`.
3. Los nombres de las variables van en español y sin acentos ni ñ: `senal`, `duracion`, `periodo`.
4. Primero resolvemos con un ciclo `for` para entender el cálculo. Después usamos la forma corta de numpy y
   comprobamos que las dos den el mismo resultado.

## Errores comunes

Cuando una celda falla, lee el último renglón del mensaje de error. Casi siempre ahí está la pista.

| El mensaje dice | Qué significa | Qué hacer |
|---|---|---|
| `NameError: name 'x' is not defined` | usas una variable que no existe | ejecuta las celdas anteriores o revisa cómo escribiste el nombre |
| `ModuleNotFoundError: funciones_pds` | Python no encuentra la biblioteca | ejecuta la primera celda del notebook |
| `IndexError: index ... is out of bounds` | pediste una posición que no existe | las posiciones van de `0` a `len(x) - 1` |
| `ValueError: operands could not be broadcast...` | operaste arreglos de distinto tamaño | revisa los tamaños con `len()` y alinea las señales (tema 05) |
| `IndentationError` | la sangría está mal | usa 4 espacios dentro de `for`, `if` y `def` |
| `TypeError: ... 'ellipsis'` | quedó un `...` sin completar | busca la marca `COMPLETA` que te falta |

## Si prefieres trabajar en tu computadora

No es necesario, pero se puede. Instala [Anaconda](https://www.anaconda.com/download), descarga este repositorio
con el botón Code > Download ZIP, descomprímelo, abre Anaconda Prompt en esa carpeta y escribe `jupyter lab`.
Si ya tienes Python instalado sin Anaconda, primero ejecuta `pip install -r requirements.txt`.
