# Clasificación multimodal de microalgas con Deep Learning

Este repositorio contiene el código desarrollado para clasificar automáticamente imágenes de microalgas mediante una red neuronal convolucional (CNN). El sistema trabaja con imágenes multicanal procedentes de Holodetect y combina etapas de lectura de datos, cálculo de características, filtrado de muestras no representativas, entrenamiento/evaluación del modelo y aplicación de un sistema de rechazo basado en confianza y características de la muestra.

Las clases consideradas son:

- `Chlorella`
- `Haematococcus`
- `Scenedesmus`

El flujo principal está implementado en `code/main.py`.

---

## Estructura del proyecto

```text
.
├── code/
│   ├── main.py
│   └── classes/
│       ├── config.py
│       ├── csv_writer.py
│       ├── data_analysis.py
│       ├── data_reader.py
│       ├── model.py
│       ├── __init__.py
│       └── saved_models/
│           └── best_model.pth
│
├── images/
│   ├── Ch/
│   ├── Ch (2)/
│   ├── Haematococcus_verde1/
│   ├── Sc/
│   ├── Scenedesmus acutus/
│   ├── generated/
│
├── data_info/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── fluorescence_summary.csv
│   └── plots/
│       ├── correlation/
│       ├── features_by_class/
│       ├── global_filtering/
│       ├── results/
│       └── unselected_features/
│
└── README.md
```

---

## Descripción de los archivos principales

### `code/main.py`

Script principal del proyecto. Ejecuta el flujo completo:

1. Fija la semilla para reproducibilidad.
2. Lee las imágenes del directorio `images/`.
3. Balancea las clases.
4. Divide los datos en entrenamiento, validación y test.
5. Calcula métricas morfológicas y de fluorescencia.
6. Analiza correlaciones y distribuciones.
7. Aplica el filtrado global.
8. Exporta los datos a CSV.
9. Entrena o carga el modelo CNN.
10. Evalúa el modelo en validación y test.
11. Calcula umbrales de confianza por clase.
12. Aplica el sistema de rechazo.
13. Genera figuras y matrices de confusión.
14. Opcionalmente, predice muestras de una carpeta externa.

---

### `code/classes/config.py`

Contiene la configuración global del proyecto.

Aquí se definen:

- Las clases del problema.
- Los nombres de las especies.
- Los canales de imagen disponibles.
- Los canales usados como entrada de la CNN.
- Las variables finales seleccionadas para el análisis y filtrado.
- El tamaño de píxel utilizado para convertir medidas a unidades físicas.
- La ruta del modelo guardado.

Ejemplo:

```python
CLASS_PREFIXES = {
    "CH": 0,
    "HA": 1,
    "SC": 2,
}

CLASS_NAMES = {
    "CH": "Chlorella",
    "HA": "Haematococcus",
    "SC": "Scenedesmus",
}

SELECTED_IMG_SUFFIXES = ["amp", "flr_1", "flr_2", "flr_3", "mask", "phase"]
```

---

### `code/classes/data_reader.py`

Se encarga de leer las imágenes y agrupar los canales pertenecientes a una misma microalga.

El lector espera que las imágenes estén organizadas por carpetas de clase. La clase real se obtiene a partir del prefijo de la carpeta principal:

- Carpetas que empiezan por `Ch` → `Chlorella`
- Carpetas que empiezan por `Ha` → `Haematococcus`
- Carpetas que empiezan por `Sc` → `Scenedesmus`

Las subcarpetas `class_*` no se usan como etiqueta real, sino únicamente como contenedores de imágenes.

Estructura esperada:

```text
images/
├── Ch/
│   ├── class_Cmicroporum/
│   │   ├── sample_amp.png
│   │   ├── sample_flu.png
│   │   ├── sample_mask.png
│   │   └── sample_phase.png
│   └── class_smallparticle/
│
├── Haematococcus_verde1/
│   └── class_*/
│
└── Scenedesmus acutus/
    └── class_*/
```

Si faltan canales individuales de fluorescencia, el programa puede generarlos a partir de la imagen compuesta `flu`.

---

### `code/classes/data_analysis.py`

Contiene las funciones de análisis y preprocesado de datos.

Entre sus tareas principales están:

- Cálculo de métricas morfológicas:
  - Área de la máscara.
  - Perímetro.
  - Circularidad.
  - Solidez.
  - Relación de aspecto.

- Cálculo de métricas de fluorescencia:
  - Fluorescencia media.
  - Proporción de área fluorescente.

- División en entrenamiento, validación y test.
- Balanceo de clases.
- Análisis de correlación.
- Generación de histogramas por clase.
- Selección de variables.
- Filtrado global de muestras anómalas.
- Visualización de muestras descartadas.
- Generación del diagrama de evaluación final.

---

### `code/classes/model.py`

Implementa el modelo de clasificación basado en CNN y todas las funciones asociadas al entrenamiento y evaluación.

Incluye:

- Construcción de la CNN.
- Lectura de datos mediante `DataLoader`.
- Entrenamiento del modelo.
- Guardado y carga de pesos.
- Evaluación en entrenamiento, validación y test.
- Matrices de confusión.
- Informes de clasificación.
- Cálculo de umbrales de confianza por clase.
- Sistema de rechazo por confianza.
- Predicción de carpetas externas.

El modelo guardado se encuentra en:

```text
code/classes/saved_models/best_model.pth
```

---

### `code/classes/csv_writer.py`

Exporta la información calculada a archivos CSV.

Genera:

- `data_info/train.csv`
- `data_info/val.csv`
- `data_info/test.csv`
- `data_info/fluorescence_summary.csv`

Estos archivos contienen las características calculadas para cada muestra y resúmenes estadísticos de las particiones.

---

## Requisitos

Se recomienda usar Python 3.10 o superior.

Librerías principales:

```text
numpy
pandas
opencv-python
matplotlib
scikit-learn
torch
torchvision
```

Si se dispone de GPU NVIDIA, se recomienda instalar PyTorch con soporte CUDA siguiendo las instrucciones oficiales de PyTorch.

---

## Cómo ejecutar el proyecto

El script debe ejecutarse desde la carpeta `code/`, ya que las rutas relativas del proyecto están definidas desde ese directorio.

```bash
cd code
python main.py
```

---

## Entrenar desde cero o usar el modelo guardado

En `main.py` existe la variable:

```python
TRAIN_MODEL = False
```

### Usar el modelo ya entrenado

```python
TRAIN_MODEL = False
```

Con esta opción se carga el modelo guardado en:

```text
classes/saved_models/best_model.pth
```

### Entrenar un modelo nuevo

```python
TRAIN_MODEL = True
```

Con esta opción el modelo se entrena desde cero, se generan las curvas de entrenamiento y se guarda el mejor modelo.

---

## Predicción sobre una carpeta externa

Al final de `main.py` se incluye una sección para predecir imágenes de una carpeta externa.

```python
PREDICT_FOLDER = True
PREDICT_FOLDER_PATH = "../../otros_datos/Scenedesmus"
```

Para desactivar esta parte:

```python
PREDICT_FOLDER = False
```

La carpeta externa debe tener una estructura similar a la del conjunto original, con subcarpetas `class_*` y los canales de imagen correspondientes.

---

## Formato esperado de las imágenes

Cada microalga puede tener varios canales de imagen. Los canales contemplados inicialmente son:

```text
amp
flr_1
flr_2
flr_3
flu
mask
phase
```

Los canales finalmente utilizados como entrada de la CNN son:

```text
amp
flr_1
flr_2
flr_3
mask
phase
```

La imagen `flu` se utiliza para extraer o verificar información de fluorescencia, pero no se usa directamente como canal de entrada del modelo final.

---

## Variables calculadas

El programa calcula distintas variables morfológicas y de fluorescencia. Las variables finales seleccionadas son:

```text
MASK_AREA
MASK_SOLIDITY
MASK_ASPECTRATIO
MEAN_FLUORESCENCE_FLU2
FLUORESCENT_AREA_RATIO_FLU2
```

Estas variables se usan principalmente para análisis, filtrado global y generación de resúmenes estadísticos.

---

## Salidas generadas

Durante la ejecución se generan archivos en el directorio `data_info/`.

### Archivos CSV

```text
data_info/train.csv
data_info/val.csv
data_info/test.csv
data_info/fluorescence_summary.csv
```

### Figuras

Las figuras se guardan en:

```text
data_info/plots/
```

Subcarpetas principales:

```text
correlation/          Matrices de correlación.
features_by_class/   Distribuciones comparativas por clase.
global_filtering/    Histogramas antes/después del filtrado global.
results/             Resultados finales del modelo.
unselected_features/ Variables calculadas pero no seleccionadas.
```

Dentro de `results/` se generan, entre otras:

```text
Confusion_matrix_val.png
Confusion_matrix_val_accepted.png
Confusion_matrix_test.png
Confusion_matrix_test_accepted.png
Test_pipeline.png
Threshold_search/
```

---

## Flujo general del sistema

```text
Lectura de imágenes
        ↓
Agrupación de canales por microalga
        ↓
Balanceo de clases
        ↓
División train / validation / test
        ↓
Cálculo de métricas morfológicas y de fluorescencia
        ↓
Análisis de correlación y selección de variables
        ↓
Filtrado global de muestras no representativas
        ↓
Entrenamiento o carga de la CNN
        ↓
Evaluación en validación y test
        ↓
Cálculo de umbrales de confianza por clase
        ↓
Sistema de rechazo
        ↓
Resultados finales
```

---

## Sistema de rechazo

Además de clasificar cada imagen, el sistema puede rechazar predicciones poco fiables.

Para ello, se calculan umbrales de confianza específicos por clase usando el conjunto de validación. Durante la inferencia, una predicción solo se acepta si la confianza del modelo supera el umbral correspondiente a la clase predicha.

Esto permite aumentar la fiabilidad de las predicciones aceptadas, a cambio de rechazar una pequeña parte de las muestras.

---

## Reproducibilidad

El script fija una semilla aleatoria para mejorar la reproducibilidad:

```python
SEED = 42
```

También se configuran opciones de PyTorch para reducir la variabilidad entre ejecuciones.

---

## Notas importantes

- Ejecutar siempre desde la carpeta `code/`.
- Las rutas del proyecto están definidas de forma relativa.
- La clase real se obtiene de la carpeta principal, no de las subcarpetas `class_*`.
- El conjunto de test solo se usa para la evaluación final.
- Los límites del filtrado global se calculan usando únicamente el conjunto de entrenamiento.
- Los umbrales de confianza se calculan usando únicamente el conjunto de validación.

---

## Autor

David Sánchez Pérez
