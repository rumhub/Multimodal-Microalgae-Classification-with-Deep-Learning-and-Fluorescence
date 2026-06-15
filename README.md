# Multimodal Microalgae Classification with Deep Learning

This repository contains the code developed to automatically classify microalgae images using a convolutional neural network (CNN). The system works with multichannel images obtained from Holodetect and combines data loading, feature extraction, filtering of non-representative samples, model training/evaluation, and a rejection system based on prediction confidence and sample characteristics.

The considered classes are:

- `Chlorella`
- `Haematococcus`
- `Scenedesmus`

The main workflow is implemented in `code/main.py`.

---

## Project structure

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
│   └── generated/
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

## Main files

### `code/main.py`

Main project script. It runs the complete workflow:

1. Sets the random seed for reproducibility.
2. Reads the images from the `images/` directory.
3. Balances the classes.
4. Splits the data into training, validation, and test sets.
5. Computes morphological and fluorescence-based metrics.
6. Analyzes correlations and feature distributions.
7. Applies global filtering.
8. Exports the data to CSV files.
9. Trains or loads the CNN model.
10. Evaluates the model on validation and test sets.
11. Computes class-specific confidence thresholds.
12. Applies the rejection system.
13. Generates figures and confusion matrices.
14. Optionally predicts samples from an external folder.

---

### `code/classes/config.py`

Contains the global configuration of the project.

It defines:

- The classes of the problem.
- The species names.
- The available image channels.
- The image channels used as CNN input.
- The final variables selected for analysis and filtering.
- The pixel size used to convert measurements to physical units.
- The saved model path.

Example:

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

Reads the images and groups the channels belonging to the same microalga.

The reader expects images to be organized by class folders. The true class is obtained from the prefix of the main folder:

- Folders starting with `Ch` → `Chlorella`
- Folders starting with `Ha` → `Haematococcus`
- Folders starting with `Sc` → `Scenedesmus`

The `class_*` subfolders are not used as the real label; they are only used as image containers.

Expected structure:

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

If individual fluorescence channels are missing, the program can generate them from the composite `flu` image.

---

### `code/classes/data_analysis.py`

Contains the data analysis and preprocessing functions.

Its main tasks include:

- Calculation of morphological metrics:
  - Mask area.
  - Perimeter.
  - Circularity.
  - Solidity.
  - Aspect ratio.

- Calculation of fluorescence metrics:
  - Mean fluorescence.
  - Fluorescent area ratio.

- Train/validation/test splitting.
- Class balancing.
- Correlation analysis.
- Generation of histograms by class.
- Feature selection.
- Global filtering of anomalous samples.
- Visualization of discarded samples.
- Generation of the final evaluation diagram.

---

### `code/classes/model.py`

Implements the CNN-based classification model and all functions related to training and evaluation.

It includes:

- CNN construction.
- Data loading using `DataLoader`.
- Model training.
- Saving and loading model weights.
- Evaluation on training, validation, and test sets.
- Confusion matrices.
- Classification reports.
- Computation of class-specific confidence thresholds.
- Confidence-based rejection system.
- Prediction on external folders.

The saved model is located at:

```text
code/classes/saved_models/best_model.pth
```

---

### `code/classes/csv_writer.py`

Exports the calculated information to CSV files.

It generates:

- `data_info/train.csv`
- `data_info/val.csv`
- `data_info/test.csv`
- `data_info/fluorescence_summary.csv`

These files contain the calculated features for each sample and statistical summaries of the data partitions.

---

## Requirements

Python 3.10 or later is recommended.

Main libraries:

```text
numpy
pandas
opencv-python
matplotlib
scikit-learn
torch
torchvision
```

If an NVIDIA GPU is available, it is recommended to install PyTorch with CUDA support following the official PyTorch installation instructions.

---

## How to run the project

The script must be executed from the `code/` folder because the project uses relative paths defined from that directory.

```bash
cd code
python main.py
```

---

## Train from scratch or use the saved model

In `main.py`, the following variable controls whether the model is trained or loaded:

```python
TRAIN_MODEL
```

### Use the pretrained model

```python
TRAIN_MODEL = False
```

With this option, the saved model is loaded from:

```text
classes/saved_models/best_model.pth
```

### Train a new model

```python
TRAIN_MODEL = True
```

With this option, the model is trained from scratch, training curves are generated, and the best model is saved.

---

## Prediction on an external folder

At the end of `main.py`, there is a section for predicting images from an external folder.

```python
PREDICT_FOLDER = True
PREDICT_FOLDER_PATH = "../../otros_datos/Scenedesmus"
```

To disable this step:

```python
PREDICT_FOLDER = False
```

The external folder must have a structure similar to the original dataset, with `class_*` subfolders and the corresponding image channels.

---

## Expected image format

Each microalga can have several image channels. The initially considered channels are:

```text
amp
flr_1
flr_2
flr_3
flu
mask
phase
```

The channels finally used as input to the CNN are:

```text
amp
flr_1
flr_2
flr_3
mask
phase
```

The `flu` image is used to extract or verify fluorescence information, but it is not used directly as an input channel of the final model.

---

## Calculated variables

The program calculates several morphological and fluorescence-based variables. The final selected variables are:

```text
MASK_AREA
MASK_SOLIDITY
MASK_ASPECTRATIO
MEAN_FLUORESCENCE_FLU2
FLUORESCENT_AREA_RATIO_FLU2
```

These variables are mainly used for analysis, global filtering, and statistical summaries.

---

## Generated outputs

During execution, files are generated inside the `data_info/` directory.

### CSV files

```text
data_info/train.csv
data_info/val.csv
data_info/test.csv
data_info/fluorescence_summary.csv
```

### Figures

Figures are saved in:

```text
data_info/plots/
```

Main subfolders:

```text
correlation/          Correlation matrices.
features_by_class/   Comparative distributions by class.
global_filtering/    Histograms before and after global filtering.
results/             Final model results.
unselected_features/ Calculated but non-selected variables.
```

Inside `results/`, the following files are generated, among others:

```text
Confusion_matrix_val.png
Confusion_matrix_val_accepted.png
Confusion_matrix_test.png
Confusion_matrix_test_accepted.png
Test_pipeline.png
Threshold_search/
```

---

## General system workflow

```text
Image reading
        ↓
Channel grouping by microalga
        ↓
Class balancing
        ↓
Train / validation / test split
        ↓
Morphological and fluorescence metric calculation
        ↓
Correlation analysis and feature selection
        ↓
Global filtering of non-representative samples
        ↓
CNN training or loading
        ↓
Validation and test evaluation
        ↓
Class-specific confidence threshold calculation
        ↓
Rejection system
        ↓
Final results
```

---

## Rejection system

In addition to classifying each image, the system can reject low-confidence predictions.

To do this, class-specific confidence thresholds are computed using the validation set. During inference, a prediction is only accepted if the model confidence is above the threshold corresponding to the predicted class.

This increases the reliability of accepted predictions at the cost of rejecting a small portion of samples.

---

## Reproducibility

The script sets a random seed to improve reproducibility:

```python
SEED = 42
```

PyTorch options are also configured to reduce variability between runs.

---

## Important notes

- Always run the project from the `code/` folder.
- Project paths are defined relatively.
- The real class is obtained from the main folder, not from the `class_*` subfolders.
- The test set is used only for the final evaluation.
- Global filtering limits are computed using only the training set.
- Confidence thresholds are computed using only the validation set.

---

## Author

David Sánchez Pérez
