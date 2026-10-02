# Project Structure and Scripts

## Environment Setup

Optimized for Linux, quite difficult to work on Windows

```bash
python3 -m venv venv_wsl
source venv_wsl/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

### Dataset Paths Configuration

To keep dataset paths secure and customizable for each local environment, the project uses a local `.env` configuration file in the project root. This file is ignored by Git to avoid leaking personal directory structures.

To configure your dataset paths:

1. Copy the template `.env_example` to a new file named `.env` in the project root:
   ```bash
   cp .env_example .env
   ```
2. Open `.env` and update the paths to point to the correct directories on your machine:
   - `DROZY_DIR`: Directory containing the DROZY dataset.
   - `ITA_PILOT_DIR`: Directory containing the ITA Pilot dataset.
   - `SEED_VIG_DIR`: Directory containing the SEED-VIG raw MAT files.
   - `SEED_VIG_LABELS`: Directory containing the SEED-VIG PERCLOS labels.

If any of these environment variables are missing when running the scripts, the program will raise a descriptive `ValueError` with setup instructions.

## Use of Logging Package

Instead of using print statements in the source code of the scalogram_cnn_project package, messages to the terminal are configured using the Logging package.

# Generator Scripts

## Unified Config-Driven Generator (`experiments/generate_scalograms.py`)

**Description**

This script unifies the scalogram generation process for both the `DROZY` and `SEED-VIG` datasets into a single CLI tool. It is fully driven by YAML configuration files placed in the `configs/dataset_generation/` directory.

It supports two modes of execution:
- **`batch`**: Generates the complete dataset (scalograms and index) based on configured subjects, sessions, and channels, and saves the output under `outputs/<output_folder>`.
- **`simple`**: Generates and saves a single test scalogram image directly under `outputs/` for parameter tuning and visualization (with the option `show_bands` to show frequency bands).

**Configuration YAML Structure**

Create config files under `configs/dataset_generation/` (e.g., `drozy_example.yaml` or `seedvig_example.yaml`). The structure is as follows:

```yaml
dataset: "DROZY" # "DROZY" or "SEED-VIG"
output_folder: "generated_scalograms_ALL_gray_overlap0.733_extra_input" # Folder name under outputs/

# Batch Mode Split Selection
subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
sessions: [1, 2, 3] # Only used for DROZY
channels: ["C3", "C4", "Cz", "Fz", "Pz"]
extra_input: true # Whether to save extra features/biomarkers in data.npy

# Parameters for scalogram generation (both modes)
scalogram_params:
  freq_min: 3
  freq_max: 30
  do_resampling: true # Only used for DROZY
  resample_freq: 128.0 # Only used for DROZY
  epoch_duration: 30.0
  overlap_ratio: 0.733
  wavelet_type: "cmor1.5-2.5"
  cmap: "gray"
  final_width_px: 64
  final_height_px: 64
  drowsiness_threshold: 4 # Only used for DROZY

# Configuration for Simple Mode (visual verification/single sample)
simple_params:
  subject: 1
  session: 1          # Only used for DROZY
  channel: "C3"
  epoch_index: 10
  show_bands: true
  final_width_px: 256  # Override resolution for high-res visualization
  final_height_px: 256
```

### Dataset Specifications (Subjects & Channels)

When configuring your batch generation YAML files, you can choose from the following subjects and EEG channels depending on the dataset selected:

#### 1. DROZY
- **Subjects:** 14 subjects total: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]`
- **Sessions:** 3 sessions total: `[1, 2, 3]`
- **EEG Channels:** `["Fz", "Cz", "C3", "C4", "Pz", "Oz"]`

#### 2. SEED-VIG
- **Subjects:** 23 subjects total: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]`
- **EEG Channels:** 17 channels total: `["FT7", "FT8", "T7", "T8", "TP7", "TP8", "CP1", "CP2", "P1", "PZ", "P2", "PO3", "POZ", "PO4", "O1", "OZ", "O2"]`

### Overlap Behavior

- **DROZY:** The DROZY generator supports an `overlap_ratio` parameter (e.g. `0.733`), which creates overlapping epoch slices of the signal. The duration step between epochs is calculated as `epoch_duration * (1 - overlap_ratio)`.
- **SEED-VIG:** The SEED-VIG generator does **not** support overlap between epochs. The signal is divided into contiguous, sequential, non-overlapping windows of `epoch_duration`. As a result, the `overlap_ratio` parameter is omitted from SEED-VIG config files.

**Execution Examples**

* **Running Batch Mode**:
  ```bash
  python3 experiments/generate_scalograms.py --config configs/dataset_generation/drozy_example.yaml --mode batch
  ```

* **Running Simple Mode**:
  ```bash
  python3 experiments/generate_scalograms.py --config configs/dataset_generation/drozy_simple_example.yaml --mode simple
  ```


# Preprocessing Approaches

**Description**

The project supports different preprocessing approaches applied to scalograms before they are used as input to the models.

**Implemented Approaches**

* **`none`**: No preprocessing is applied. The scalograms are used as they are generated.
* **`rpca_isolated`**: Robust Principal Component Analysis (RPCA) is applied to each scalogram individually. It decomposes the image into a low-rank matrix (L) and a sparse matrix (S).
* **`rpca_juxtaposed`**: RPCA is applied to a set of scalograms from different channels that are horizontally concatenated (juxtaposed) for the same epoch.

> **Important Note:** The `rpca_juxtaposed` approach **does not support the `separate` mode** for model runners. If used with `rpca_juxtaposed`, the mode will be automatically changed to the `mix` mode.


# RPCA Preprocessing Scripts

**Description**

These scripts apply Robust Principal Component Analysis (RPCA) to generated scalograms, separating them into a low-rank component (L) and a sparse component (S). They are executed via command line, accepting arguments to define the input, output, and RPCA parameters.

**Scripts**

## `experiments/apply_rpca_isolated.py`

Applies RPCA to each scalogram individually in a given folder.

**CLI Arguments:**
- `--input_folder`: Full path of the input folder containing scalograms (default: `data/generated_scalograms_ALL_gray_overlap0.733_extra_input_example`).
- `--output_folder`: Output folder name created inside `outputs/` (default: `isolated_scalograms`).
- `--cmap`: Colormap to use (default: `gray`).
- `--lamb`: RPCA lambda parameter (default: `None` for default value).
- `--mu`: RPCA mu parameter (default: `None` for default value).
- `--tolerance`: RPCA tolerance parameter (default: `None` for default value).
- `--max_iteration`: RPCA max iteration parameter (default: `None` for default value).

**Execution Example:**
```bash
python3 experiments/apply_rpca_isolated.py \
    --input_folder="data/generated_scalograms_ALL_gray_overlap0.733_extra_input_example" \
    --output_folder="isolated_scalograms_custom" \
    --lamb=0.15
```

## `experiments/apply_rpca_juxtaposed.py`

Applies RPCA to horizontally juxtaposed scalograms from different channels for the same epoch.

**CLI Arguments:**
- `--input_folder`: Full path of the input folder containing scalograms (default: `data/generated_scalograms_ALL_gray_overlap0.733_extra_input_example`).
- `--output_folder`: Output folder name created inside `outputs/` (default: `juxtaposed_scalograms`).
- `--cmap`: Colormap to use (default: `gray`).
- `--lamb`: RPCA lambda parameter (default: `None` for default value).
- `--mu`: RPCA mu parameter (default: `None` for default value).
- `--tolerance`: RPCA tolerance parameter (default: `None` for default value).
- `--max_iteration`: RPCA max iteration parameter (default: `None` for default value).

**Execution Example:**
```bash
python3 experiments/apply_rpca_juxtaposed.py \
    --input_folder="data/generated_scalograms_ALL_gray_overlap0.733_extra_input_example" \
    --output_folder="juxtaposed_scalograms_custom" \
    --lamb=0.15
```

## `experiments/apply_rpca_simple.py`

Applies RPCA to a single image to test multiple lambda parameters. Ideal for parameter tuning and visualizing results.

**CLI Arguments:**
- `--image_path`: Path to the test image (default: `data/generated_scalograms_ALL_gray_overlap0.733_extra_input_example/img_0a85796bce.png`).
- `--output_folder`: Output folder name created inside `outputs/` (default: `rpca_simple_output`).
- `--lambdas`: Space-separated list of RPCA lambda parameters to test (default: `0.05 0.10 0.125 0.15 0.175 0.2 0.25 0.30`).
- `--mu`: RPCA mu parameter (default: `None` for default value).
- `--tolerance`: RPCA tolerance parameter (default: `None` for default value).
- `--max_iteration`: RPCA max iteration parameter (default: `None` for default value).
- `--cmap`: Colormap to use (default: `gray`).

**Execution Example:**
```bash
python3 experiments/apply_rpca_simple.py \
    --image_path="data/generated_scalograms_ALL_gray_overlap0.733_extra_input_example/img_0a85796bce.png" \
    --output_folder="rpca_simple_custom" \
    --lambdas 0.125 0.15 0.175
```

# Models

**Description**

Models are function that create model and callback objects.

**Versions**

### v0
It is a model with fixed hyperparameters; its architecture matches the description of two-layered CNN-2D as described by A. Zayed (2025). The required parameters are:

```python
REQUIRED_TRAIN_KEYS = ["seed", "optimizer_name", "batch_size", "subjects", "overlap", "learning_rate", "label_smoothing", "num_epochs"]

REQUIRED_MODEL_KEYS = ["channels", "epsilon", "momentum", "cmap", "mode", "from_logit", "final_width_px", "final_height_px", "preprocessing"]
```


### v1
It is a model with variable hyperparameters; its architecture is a variation of the one proposed by A. Zayed (2025), as it allows the user to utilize one extra convolutional layers, as well as adjust the number of filters in each layer, the kernel size etc. The required parameters are:

```python
REQUIRED_TRAIN_KEYS = ["seed", "optimizer_name", "batch_size", "subjects", "overlap", "learning_rate"]

REQUIRED_MODEL_KEYS = ["channels", "epsilon", "momentum", "cmap", "mode", "n_additional_features",
                        "kernel_size", "extra_layer", "extra_layer_num_filters", "num_neurons_dense",
                        "first_layer_num_filters", "second_layer_num_filters", "final_width_px", "final_height_px", "preprocessing"]
```

### v2
It is also a model with variable hyperparameters; its architecture is a variation of the one proposed by A. Zayed (2025), as it allows the user to utilize one extra convolutional layers, as well as adjust the number of filters in each layer, the kernel size etc. Also, right after the flatten layer, the CNN receives extra input, which are normalized biomarkers calculated as the ratio of power in different bands. The required parameters are:

```python
REQUIRED_TRAIN_KEYS = ["seed", "optimizer_name", "batch_size", "subjects", "overlap", "learning_rate"]

REQUIRED_MODEL_KEYS = ["channels", "epsilon", "momentum", "cmap", "mode", "n_additional_features",
                        "kernel_size", "extra_layer", "extra_layer_num_filters", "num_neurons_dense",
                        "first_layer_num_filters", "second_layer_num_filters", "final_width_px", "final_height_px", "preprocessing"]
```

# Model Runners

**Description**

A model runner loads data from memory and, with a model that it receives as parameter, runs a training/validation session. For the model runners, there are two options for dealing with data: separate and mix. The first one differentiate between channels, i.e., its input are the stack of color maps of diffente channels given a specific epoch. On the other hand, the "mix" option presupposes that all scalograms come from the same channel (which can suprisingly yield good results).

**Versions**

### v0
It is prepared to receive scalograms from a selected set of channels, using a color map to the user's choice. It suffers from data leakage, due to the the overlap between the epochs considered.

### v1
It is also prepared to receive scalograms from a selected set of channels, using a color map to the user's choice. It solves the problem of data leakage by destinating the first seven minutes of each session to training and the rest to testing.

### v2
It is also prepared to receive scalograms from a selected set of channels, using a color map to the user's choice. It solves the problem of data leakage by applying a Leave-One-Subject-Out (LOSO) validation.

# Experiment Scripts Overview

This repository contains several scripts used to run experiments with CNN models trained on scalogram images. The scripts support three main experiment strategies:

* *Leave-One-Subject-Out cross-validation*
* *Manual grid search*
* *Automated hyperparameter optimization using Keras Tuner*

Each script orchestrates model creation, training execution, parameter management, and experiment reproducibility. 

These scripts are now executed via command line, accepting arguments to define the input, output, model, and parameters. To ensure the scripts restart automatically in case of memory leaks or crashes, it is highly recommended to run them using the provided `run_until_it_ends.sh` wrapper.

Here are the details and execution examples for each script.

# 1. `run_cross_validation_loso.py`

This script performs *Leave-One-Subject-Out (LOSO) cross-validation* defined in a YAML configuration file. It has only support for fixed and choice modes.

**Execution Example:**
```bash
./run_until_it_ends.sh experiments/run_cross_validation_loso.py \
    --input_folder="generated_scalograms_ALL_gray_overlap0.733_extra_input_example" \
    --output_folder="generic_loso_example" \
    --model="v1" \
    --params_file="cross_validation_loso_example.yaml"
```

# 2. `run_gridsearch.py`

This script performs a *grid search* over the hyperparameter space defined in a YAML configuration file. It has only support for fixed and choice modes.

**CLI Arguments:**
- `--input_folder`: Name of the folder under `outputs/` containing the dataset.
- `--output_folder`: Output folder name to be created under `outputs/`.
- `--params_file`: YAML configuration file with the parameter grid.
- `--model`: Model version to use (`v0`, `v1`, or `v2`).
- `--model_runner`: Model runner version to use (`v0`, `v1`, or `v2`).
- `--force-cpu`: Optional flag to force CPU execution (disabling GPUs to prevent VRAM allocation crashes).

**Execution Example:**
```bash
./run_until_it_ends.sh experiments/run_gridsearch.py \
    --input_folder="generated_scalograms_ALL_gray_overlap0.733_extra_input_example" \
    --output_folder="generic_gridsearch_example" \
    --model="v1" \
    --model_runner="v1" \
    --params_file="gridsearch_example.yaml" \
    --force-cpu
```

# 3. `run_keras_tuner.py`

The script performs a *random search* over the hyperparameter space defined in a YAML configuration file. It has support for all modes, including the interval ones.

**Execution Example:**
```bash
./run_until_it_ends.sh experiments/run_keras_tuner.py \
    --input_folder="generated_scalograms_ALL_gray_overlap0.733_extra_input_example" \
    --output_folder="generic_keras_example" \
    --model="v1" \
    --model_runner="v1" \
    --max_trials=100 \
    --params_file="keras_search_example.yaml"
```

## YAML Parameter Loading

The hyperparameter search space is loaded dynamically:

```python
with open(PARAMS_FILE) as f:
    config_params = yaml.safe_load(f)
```

The YAML file defines:

* `MODEL_HYPER_PARAMS`
* `MODEL_TRAIN_PARAMS`

These parameters are interpreted by the `build_model()` function.

---

## Model Builder

The function:

```
build_model()
```

translates YAML parameter definitions into Keras Tuner search parameters.

It dynamically constructs the model and optimizer based on the sampled hyperparameters.


## Trial Execution

For each trial, the tuner:

1. Samples hyperparameters
2. Builds the model
3. Runs training
4. Reports the validation loss.


## Model Saving

Each trained model is saved automatically in:

```
saved_models/trial_<id>/model.keras
```

This allows inspection and reuse of trained models.


## Search Configuration

The number of experiments performed by the tuner is controlled by:

```
MAX_TRIALS
```

## Experiment Reproducibility

For reproducibility, the YAML configuration used for the search is copied to the experiment output directory:

```
search_params.yaml
```

## Final Results

After the search finishes, the best trials are stored in:

```
best_trials.txt
```

Example entry:

```
Rank 0 | val_loss=0.21543 | params={...}
```



# YAML Configuration for the experiments scripts

This project uses `.yaml` configuration files to define both *model hyperparameters* and *training parameters* used during hyperparameter optimization.

These parameters are interpreted by the `build_model()` module, which dynamically constructs a search space for *Keras Tuner*.

The configuration is divided into two main sections:

- `MODEL_HYPER_PARAMS` → parameters that affect the *architecture of the model*
- `MODEL_TRAIN_PARAMS` → parameters that affect *training behavior*

Each parameter specifies *how it should be sampled* during the search.


# Configuration Structure

Example:

```yaml
MODEL_HYPER_PARAMS:

  epsilon:
    mode: log_interval
    values: [1e-4, 1e-2]

MODEL_TRAIN_PARAMS:

  learning_rate:
    mode: log_interval
    values: [1e-5, 1e-2]
````

Each parameter must contain the following structure:

```yaml
parameter_name:
  mode: <sampling_mode>
  values: <parameter_values>
```


# Parameter Sampling Modes

The `mode` field defines *how the parameter will be sampled* during hyperparameter search.

The following modes are supported.


## 1. fixed

The parameter value is constant and *not optimized*.

Example:

```yaml
cmap:
  mode: fixed
  values: ["gray"]
```

Use this when the parameter *must remain constant across experiments*.


## 2. choice

The parameter is chosen from a *discrete set of values*.

Example:

```yaml
optimizer_name:
  mode: choice
  values: ["adam", "sgd", "rmsprop"]
```

Use this when you want to test *different categorical options*.

Typical examples include:

* optimizer
* activation functions
* architecture variants


## 3. float_interval

Samples a *continuous floating-point value* within an interval.

Example:

```yaml
momentum:
  mode: float_interval
  values: [0.85, 0.99]
```

Use this when the parameter is *continuous* and does *not require logarithmic scaling*.


## 4. log_interval

Samples a floating-point value *logarithmically*.

Example:

```yaml
learning_rate:
  mode: log_interval
  values: [1e-5, 1e-2]
```


Use this for parameters that vary across *orders of magnitude*, such as:

* learning rate
* epsilon
* regularization coefficients


## 5. int_interval

Samples an *integer value within a range*.

Example:

```yaml
batch_size:
  mode: int_interval
  values: [16, 128]
```

Typical uses include:

* batch size
* number of neurons
* number of filters
* kernel size


# Ready-for-test yaml files

In the directory: 

```
/configs/hyperparameter_search
```

One can find examples of `.yaml` files for each experiment script.

Additionally, in the directory:

```
/configs/dataset_generation
```

one can find example `.yaml` configuration files for generating scalograms (both in batch and simple modes) for the DROZY and SEED-VIG datasets.

# RNN Signal Forecasting and Reconstruction

**Description**

This module provides a pipeline to forecast physiological/EEG signals into subsequent future timesteps or minutes (signal reconstruction) using Recurrent Neural Networks (RNNs) in Keras/TensorFlow. The pipeline is designed to load signals in a database-agnostic manner, train forecasting models using strict chronological splits to avoid data leakage, reconstruct continuous time series using freshest-window aggregation, and evaluate fidelity using correlation, amplitude, and error metrics.


---

## 1. Project Organization

### Data Abstraction (`utils/`)
* **`src/scalogram_cnn_project/utils/signal_data.py`**: Contains `SignalData`, a unified class storing 2D raw signal time-series, channel names, and sampling frequency. It provides helper methods to extract individual channels, resample frequencies, and slice specific time windows.
* **`src/scalogram_cnn_project/utils/signal_loader.py`**: Contains `SignalLoader`, exposing static methods to load SEED-VIG (`.mat` structs) and DROZY (`.edf` via MNE) signal files and parse them into standardized `SignalData` objects.
* **`src/scalogram_cnn_project/utils/plot_results.py`**: Formats and draws high-resolution dual-panel comparison plots (panorama + 60-second zoom) displaying ground truth, prediction, and fidelity metrics.

### Recurrent Model Architectures (`models_for_prediction/`)
* **`model_predict_v0.py`**: Implements an **LSTM Direct Projection** architecture, mapping an input sequence to future horizon steps using an LSTM recurrent backbone and a dense projection layer.
* **`model_predict_v1.py`**: Implements a **GRU Direct Projection** architecture, mapping an input sequence to future horizon steps using a gated recurrent unit backbone.
* **`model_predict_builder.py`**: Factory function to dynamically instantiate and compile prediction models based on a version code (`v0` for LSTM, `v1` for GRU), optimizer (`adam`, `rmsprop`, `sgd`), and learning rate.

---

## 2. Parameter Configurations (`configs/model_training_rnn_predictor/`)

Configuration templates for prediction models are placed under `/configs/model_training_rnn_predictor/` or `/configs/hyperparameter_search_rnn/`.

Example configuration:
```yaml
dataset_type: "seed_vig"
channel: "CP2"
subject: 1                # Subject ID to filter training files (only one subject at a time)
model_version: "v0"       # "v0" for LSTM, "v1" for GRU
optimizer_name: "rmsprop" # "adam", "rmsprop", or "sgd"

# Temporal Window Definition (supports minutes, seconds, or exact discrete timesteps):
# Option A: Duration in minutes
# input_min: 5.0
# predict_min: 2.0
# stride_sec: 30.0

# Option B: Duration in seconds
# input_sec: 30.0
# predict_sec: 30.0
# stride_sec: 1.0

# Option C: Exact discrete timesteps (e.g., 1-step ahead forecasting at 100 Hz = 10 ms horizon)
input_steps: 100          # 1.0 second past context at 100 Hz
predict_steps: 1          # 1 step ahead (10 ms horizon)
stride_steps: 1           # Stride of 1 sample

epochs: 10
batch_size: 64
latent_dim: 32
learning_rate: 0.001
train_split: 0.8          # Chronological train/validation fraction (first 80% train, last 20% test)
resample_freq: 100.0      # Resampling frequency in Hz (e.g. 100.0 Hz to retain Beta/Gamma dynamics)
force_cpu: false          # Set to true to force CPU execution and avoid GPU VRAM allocation crashes
max_train_samples: null   # Optional cap on training samples for rapid prototyping
save_plot: true           # Set to false to disable PNG rendering (faster execution)
output_plot: null         # Path to save high-resolution comparison plot (or null for default)
output_model: null        # Path to save trained model weights
```

---

## 3. Core Methodologies and Execution

### A. Data Leakage Prevention & Z-Score Normalization
To guarantee zero data contamination and ensure numerical stability:
1. **Chronological Partitioning**: The continuous signal is strictly split chronologically (e.g., the first 80% for training and the remaining 20% for validation/testing).
2. **Transition Gap Rejection**: To avoid temporal leakage from sliding windows overlapping the train/test boundary, a gap is discarded:
   $$\text{neglected\_windows} = \lceil \frac{T_{in} + T_{out}}{\text{stride}} \rceil$$
3. **No Leakage $z$-Score Scaling**: Normalization statistics ($\mu_{\text{train}}$, $\sigma_{\text{train}}$) are computed **strictly on the training partition**. The test partition is standardized using the exact same $(\mu_{\text{train}}, \sigma_{\text{train}})$.
4. **Physical Scale Invariance**: Model predictions in standardized units are mapped back to original physical units ($\mu\text{V}$) via $y_{\text{phys}} = \hat{y} \cdot \sigma_{\text{train}} + \mu_{\text{train}}$, avoiding precision underflow while preserving physical voltage scales.

### B. Signal Reconstruction Strategy (Shortest Horizon / Freshest Prediction)
When reconstructing the continuous validation signal across overlapping sliding windows:
* **Why Average-Based Merging Flattened Signals**: Arithmetic averaging across multi-step overlapping predictions causes destructive phase cancellation in oscillatory signals, artificially flattening predictions toward zero.
* **Shortest Horizon / Freshest Prediction Selection (Option 1)**: For every temporal point $t$ in the test series, the reconstruction pipeline selects the prediction generated by the **most recent input window** (i.e. the smallest prediction horizon $h = t - t_{\text{start}} + 1$), discarding older multi-step projections. This preserves authentic amplitude, phase, and peak dynamics without damping.

### C. Evaluation and Fidelity Metrics

Every candidate in training and grid search logs six core metrics to `progress.json` and `results.jsonl`:

| Metric Name | Mathematical Definition | Biological & Practical Significance |
| :--- | :--- | :--- |
| **`loss`** | $\text{MSE}_{\text{train}} = \frac{1}{B} \sum_{i=1}^B (y_i - \hat{y}_i)^2$ | **Training Loss:** Mean squared error on the training partition (first 80% of the signal), accumulated across batches during the final epoch. |
| **`val_loss`** | $\text{MSE}_{\text{val}}$ | **Validation Loss:** Loss evaluated on the unobserved validation split (last 20% of the signal). Equivalent to `val_mse`. |
| **`val_mse`** | $\frac{1}{N} \sum_{t=1}^N (y_t - \hat{y}_t)^2$ | **Mean Squared Error:** Measures residual variance in $z$-score units ($\sigma=1$). A value $< 0.01$ indicates residual variance $< 1\%$ of total signal variance. |
| **`val_mae`** | $\frac{1}{N} \sum_{t=1}^N \|y_t - \hat{y}_t\|$ | **Mean Absolute Error:** Average point-to-point absolute error in standard deviation units. A `val_mae` of $0.073$ means predictions deviate by only $7.3\%$ of the signal's standard deviation. |
| **`val_pearson_corr`** | $r = \frac{\sum (y_t - \bar{y})(\hat{y}_t - \bar{\hat{y}})}{\sqrt{\sum (y_t - \bar{y})^2 \sum (\hat{y}_t - \bar{\hat{y}})^2}}$ | **Pearson Correlation ($r$):** Evaluates waveform shape, peak synchronization, and temporal phase alignment. Values $> 0.98$ represent near-perfect phase tracking. |
| **`val_amp_ratio`** | $\text{Ratio} = \frac{\sigma_{\text{pred}}}{\sigma_{\text{gt}}}$ | **Amplitude Tracking Ratio:** Directly diagnoses the flat-curve issue. Values $\approx 1.0$ ($> 0.95$) confirm that the model preserves full oscillatory wave height without damping to zero. |

> [!NOTE]
> **Why is `val_loss` frequently lower than `loss`?**  
> 1. **Batch Accumulation vs. End-of-Epoch Evaluation:** In Keras, `loss` is the running average across all mini-batches throughout the epoch (including early steps before weight updates). `val_loss` is computed strictly *after* all weight updates have finished.  
> 2. **Physiological Signal Stationarity:** In the SEED-VIG dataset, the first 80% (training) contains active vigilance state transitions and motion/blink artifacts (higher entropy/variance). The final 20% (validation) often represents stable, synchronized drowsiness dynamics (dominated by regular Alpha/Theta rhythms), which possess higher predictability and lower residual error.  
> 3. **Healthy Generalization:** A lower validation loss confirms zero overfitting.

### D. Training the RNN Forecaster
Run `experiments/train_rnn.py` via CLI or YAML configuration:

* **Training via YAML config**:
  ```bash
  python3 experiments/train_rnn.py --config configs/hyperparameter_search_rnn/forecast_grid_6_1step_100hz.yaml
  ```

* **Training via 1-Step CLI overrides with RMSProp**:
  ```bash
  python3 experiments/train_rnn.py \
      --dataset-type seed_vig \
      --channel CP2 \
      --model-version v0 \
      --subject 1 \
      --input-steps 100 \
      --predict-steps 1 \
      --stride-steps 1 \
      --resample-freq 100.0 \
      --optimizer-name rmsprop \
      --learning-rate 0.001 \
      --latent-dim 32 \
      --batch-size 64 \
      --epochs 10 \
      --train-split 0.8 \
      --output-plot outputs/test_1step_plot.png
  ```

### E. Running Predictions (run_pipeline)

Run `experiments/run_pipeline.py` to load a signal, extract an input window, perform the RNN forecast, and save the outputs to the outputs folder.

* **Running the pipeline**:
  ```bash
  python3 experiments/run_pipeline.py \
      --file 10_20151125_noon.mat \
      --dataset-type seed_vig \
      --channel O1 \
      --start-min 2.0 \
      --end-min 7.0 \
      --predict-min 2.0 \
      --model-path outputs/models/rnn_predict_v0_seed_vig_O1.h5
  ```

This command will output:
1. **A reconstructed future signal** saved to a MATLAB `.mat` file (e.g. `outputs/predicted_10_20151125_noon_O1.mat`).
2. **A dual-panel comparison plot** (Panorama + 60s Zoom) saved to `outputs/plot_10_20151125_noon_O1.png`.

---

# RNN-MLP Sleepiness Classification Pipeline

**Description**

This module implements a Two-Stage coupled architecture where a Multi-Layer Perceptron (MLP) binary classifier is stacked directly on top of a frozen pre-trained RNN forecaster model. The coupled model classifies whether the subject is alert or drowsy based on temporal signal windows.

Training is performed on standard-scaled inputs using Binary Crossentropy loss, and evaluated with **Accuracy** as the final metric.

---

## 1. Classification Configuration (`configs/model_training_rnn_classifier/`)

Create configuration files under `configs/model_training_rnn_classifier/`:

* **SEED-VIG Configuration** (e.g. `configs/model_training_rnn_classifier/seedvig_classify_example.yaml`):
  ```yaml
  dataset_type: "seed_vig"
  channel: "CP2"
  lead_time_sec: 30.0     # Anticipation lead time (target PERCLOS state at t + 30s)
  stride_sec: 5.0         # 5.0-second sliding stride
  rnn_model_path: "outputs/models/best_rnn_predictor_seedvig_CP2.h5"  # Pre-trained RNN forecaster model (.h5)
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
  train_split: 0.8
  drowsiness_threshold: 0.5   # PERCLOS threshold for driver drowsiness (matches round(perclos))
  class_weight_mode: "balanced"  # Handle class imbalance: "balanced", "none", or custom
  save_plot: true             # Generate comparative Loss & Accuracy training curves
  output_plot: null           # Path to save comparative history plot (or null for default)
  output_model: null
  ```

* **DROZY Configuration** (e.g. `configs/model_training_rnn_classifier/drozy_classify_example.yaml`):
  ```yaml
  dataset_type: "drozy"
  channel: "C3"
  subjects: [1, 2, 3]
  lead_time_sec: 0.0
  stride_sec: 5.0
  rnn_model_path: "outputs/models/best_rnn_predictor_seedvig_CP2.h5"  # Path to pre-trained model (.h5)
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
  train_split: 0.8
  drowsiness_threshold: 4  # KSS threshold for DROZY dataset
  class_weight_mode: "balanced"
  save_plot: true
  output_plot: null
  output_model: null
  ```

---

## 2. Executing Training and Generating History Plots

Run `experiments/train_rnn_classifier.py` to couple the pre-trained RNN latent backbone, train the MLP classification layers, and automatically generate training/validation curves.

* **Training via YAML config**:
  ```bash
  python3 experiments/train_rnn_classifier.py --config configs/model_training_rnn_classifier/seedvig_classify_example.yaml
  ```

* **Training with Anticipation Lead Time ($X = 30\text{s}$), Balanced Class Weights, and Plot Output**:
  ```bash
  python3 experiments/train_rnn_classifier.py \
      --dataset-type seed_vig \
      --channel CP2 \
      --lead-time-sec 30.0 \
      --rnn-model-path outputs/models/best_rnn_predictor_seedvig_CP2.h5 \
      --class-weight-mode balanced \
      --epochs 10 \
      --batch-size 32 \
      --output-plot outputs/classifier_training_history.png
  ```

### Handling Class Imbalance
The classification pipeline provides three ways to counter class imbalance (e.g. ~65.3% Alert vs 34.7% Drowsy in SEED-VIG):
1. `--class-weight-mode balanced` (or in YAML `class_weight_mode: "balanced"`): Automatically computes inversely proportional weights using `sklearn.utils.class_weight.compute_class_weight`.
2. `--drowsy-weight <float>` (e.g. `--drowsy-weight 2.0`): Sets Alert weight to 1.0 and Drowsy weight to the specified float.
3. `--class-weight '<json>'`: Passes a custom dictionary directly (e.g. `--class-weight '{"0": 1.0, "1": 1.88}'`).

### Training History Plots
By default (`--save-plot`, `default=True`), training outputs a comparative plot with two side-by-side subplots:
* **Loss Curve:** Binary Cross-Entropy loss over epochs (Train vs. Validation).
* **Accuracy Curve:** Classification Accuracy over epochs (Train vs. Validation).
* **Integer Epoch Axis:** The x-axis uses strictly discrete integer values ($1, 2, \dots, N$) representing full training epochs.
* To disable plot generation, pass `--no-save-plot` (or `save_plot: false` in YAML).

---

## 3. Running Integration Tests

To run the integration tests verifying the classification pipeline functionality:
```bash
python3 -m unittest tests/test_rnn_classification.py
```

---

# RNN Hyperparameter Grid Search (Temporal Split)

**Description**

This module provides wrappers (`run_rnn_gridsearch.py` and `run_rnn_classifier_gridsearch.py`) to perform hyperparameter grid search using **Temporal Intra-Session Chronological Splitting** (`train_split: 0.8` — the first 80% of time for training, last 20% for validation within the same session/subjects).

> [!NOTE]
> **Temporal Split vs. LOSO:** These scripts evaluate time-based generalization within the specified subject(s). For cross-subject evaluation where each candidate is evaluated across unseen individuals, see [LOSO Hyperparameter Grid Search](#loso-hyperparameter-grid-search).

Each candidate combination runs inside an isolated subprocess to prevent VRAM memory leaks or GPU out-of-memory errors in TensorFlow, communicating final validation metrics via a temporary JSON file.

---

## 1. Configurations (`configs/hyperparameter_search_rnn/`)

Parameter spaces are defined under `configs/hyperparameter_search_rnn/`:

* **1-Step 100 Hz RNN Forecasting Search (Temporal Split)** (e.g. `configs/hyperparameter_search_rnn/forecast_grid_6_1step_100hz.yaml`):
  ```yaml
  MODEL_HYPER_PARAMS:
    model_version:
      mode: "choice"
      values: ["v0", "v1"]       # LSTM vs GRU
    latent_dim:
      mode: "choice"
      values: [16, 32, 64]
  MODEL_TRAIN_PARAMS:
    optimizer_name:
      mode: "choice"
      values: ["adam", "rmsprop"]
    learning_rate:
      mode: "choice"
      values: [0.001, 0.0001]
    epochs:
      mode: "choice"
      values: [10, 20]
    batch_size:
      mode: "choice"
      values: [64, 128]
    dataset_type:
      mode: "fixed"
      values: ["seed_vig"]
    channel:
      mode: "fixed"
      values: ["CP2"]
    subject:
      mode: "fixed"
      values: [1]
    resample_freq:
      mode: "fixed"
      values: [100.0]
    input_steps:
      mode: "fixed"
      values: [100]
    predict_steps:
      mode: "fixed"
      values: [1]
    stride_steps:
      mode: "fixed"
      values: [1]
    train_split:
      mode: "fixed"
      values: [0.8]
  ```

* **RNN Coupled Classification Search (Temporal Split)** (e.g. `configs/hyperparameter_search_rnn/seedvig_anticipatory_classifier_grid_1.yaml`):
  ```yaml
  MODEL_HYPER_PARAMS:
    learning_rate:
      mode: "choice"
      values: [0.01, 0.001, 0.0005]
  MODEL_TRAIN_PARAMS:
    epochs:
      mode: "choice"
      values: [10, 20]
    batch_size:
      mode: "choice"
      values: [32, 64]
    class_weight_mode:
      mode: "fixed"
      values: ["none"]
    lead_time_sec:
      mode: "choice"
      values: [0.0, 5.0, 30.0]
    stride_sec:
      mode: "fixed"
      values: [5.0]
    train_split:
      mode: "fixed"
      values: [0.8]
    dataset_type:
      mode: "fixed"
      values: ["seed_vig"]
    channel:
      mode: "fixed"
      values: ["CP2"]
    resample_freq:
      mode: "fixed"
      values: [100.0]
    rnn_model_path:
      mode: "fixed"
      values: ["outputs/models/best_rnn_predictor_seedvig_CP2.h5"]
  ```

---

## 2. Executing Temporal Split Grid Search

* **RNN Forecasting Grid Search (Temporal Split)**:
  ```bash
  python3 experiments/run_rnn_gridsearch.py \
      --output_folder forecast_grid_6 \
      --params_file configs/hyperparameter_search_rnn/forecast_grid_6_1step_100hz.yaml \
      --force-cpu
  ```

* **RNN Coupled Classification Grid Search (Temporal Split)**:
  ```bash
  python3 experiments/run_rnn_classifier_gridsearch.py \
      --output_folder rnn_classifier_search \
      --params_file configs/hyperparameter_search_rnn/seedvig_anticipatory_classifier_grid_1.yaml \
      --force-cpu
  ```

---

# LOSO Hyperparameter Grid Search

**Description**

`experiments/run_rnn_classifier_loso_gridsearch.py` performs hyperparameter exploration under **Leave-One-Subject-Out (LOSO) Cross-Validation**. 

For **each candidate hyperparameter combination**, the runner executes a complete cross-validation across all designated subjects (e.g. subjects $1 \dots 5$ or all 23 SEED-VIG subjects), computing the Global Validation Accuracy ($\mu \pm \sigma$) and Global Validation Loss ($\mu \pm \sigma$).

### 1. Configuration (`configs/hyperparameter_search_rnn/seedvig_loso_classifier_grid_example.yaml`)

```yaml
MODEL_HYPER_PARAMS:
  learning_rate:
    mode: "choice"
    values: [0.001, 0.0005]
  model_version:
    mode: "choice"
    values: ["v0", "v1"]
  hidden_units_1:
    mode: "choice"
    values: [64, 128]
  hidden_units_2:
    mode: "fixed"
    values: [32]
  dropout_1:
    mode: "choice"
    values: [0.3, 0.4]
  dropout_2:
    mode: "fixed"
    values: [0.2]
  optimizer_name:
    mode: "fixed"
    values: ["adam"]

MODEL_TRAIN_PARAMS:
  lead_time_sec:
    mode: "choice"
    values: [0.0, 5.0, 30.0]
  epochs:
    mode: "fixed"
    values: [15]
  batch_size:
    mode: "choice"
    values: [32, 64]
  class_weight_mode:
    mode: "fixed"
    values: ["none"]
  fine_tune_rnn:
    mode: "choice"
    values: [false, true]
  stride_sec:
    mode: "fixed"
    values: [5.0]
  dataset_type:
    mode: "fixed"
    values: ["seed_vig"]
  channel:
    mode: "fixed"
    values: ["CP2"]
  resample_freq:
    mode: "fixed"
    values: [100.0]
  drowsiness_threshold:
    mode: "fixed"
    values: [0.5]
  rnn_model_path:
    mode: "fixed"
    values: ["outputs/models/best_rnn_predictor_seedvig_CP2.h5"]
  subjects:
    mode: "fixed"
    values:
      - [1, 2, 3, 4, 5]
```

### 2. Executing LOSO Grid Search

```bash
python3 experiments/run_rnn_classifier_loso_gridsearch.py \
    --params_file configs/hyperparameter_search_rnn/seedvig_loso_classifier_grid_example.yaml \
    --output_folder rnn_classifier_loso_gridsearch
```

* **Overriding Evaluated Subjects via CLI**:
  ```bash
  python3 experiments/run_rnn_classifier_loso_gridsearch.py \
      --params_file configs/hyperparameter_search_rnn/seedvig_loso_classifier_grid_example.yaml \
      --output_folder rnn_classifier_loso_gridsearch \
      --subjects 1 2 3
  ```

### 3. LOSO Grid Search Outputs

Inside `outputs/<output_folder>/`:
* **`results.jsonl`**: Registry ranking each candidate by `mean_val_accuracy` ($\mu \pm \sigma$) and `mean_val_loss`.
* **`cand_XXXXX_<hash_id>/`**: Per-candidate folders containing:
  - `loso_summary.json`: Detailed breakdown of all folds for that candidate.
  - `loso_evolution_overview.png`: Multi-panel plot (learning dynamics + subject bar chart) for that candidate.
  - `fold_subj_XX/`: Model weights (`.h5`) and training curves for each individual fold.
* **`progress.json` & `param_registry.json`**: Crash-resilient state trackers.

---

# Leave-One-Subject-Out (LOSO) Cross-Validation

**Description**

Leave-One-Subject-Out (LOSO) cross-validation evaluates how effectively the coupled RNN-MLP anticipatory drowsiness classifier generalizes to completely unseen subjects.

### Why LOSO with Full Signal?
In driving/fatigue experiments (such as SEED-VIG and DROZY), vigilance levels decrease monotonically over the course of a driving session. When performing intra-session chronological splits (e.g. first 80% train / last 20% validation), the final 20% of a session often contains **100% drowsy labels** (e.g. PERCLOS $\ge 0.5$). Consequently, an intra-session model predicting the majority class in the final minutes would yield an artificially inflated validation accuracy of 100%.

Under **LOSO Cross-Validation with Full Continuous Signals** (`--use-full-signal` and `--validation-subject`):
- All continuous EEG samples from the training subjects (e.g. 22 subjects in SEED-VIG) are combined for training.
- 100% of the continuous EEG recording from the left-out subject (e.g. Subject $k$) is used strictly for validation.
- Per-file z-score standardization is applied individually to prevent cross-session leakage.
- The resulting evaluation measures true inter-subject physiological generalization.

---

## 1. Single Fold Execution (`experiments/train_rnn_classifier.py`)

You can train a single LOSO fold directly by passing `--validation-subject <ID>` (or `--validation-subjects`):

```bash
python3 experiments/train_rnn_classifier.py \
    --config configs/loso_rnn_classifier/seedvig_loso_v0_cp2.yaml \
    --validation-subject 1 \
    --output-model outputs/model_loso_s01.h5 \
    --output-plot outputs/history_loso_s01.png
```

* `--validation-subject`: Integer ID of the left-out subject (e.g. `1`).
* `--use-full-signal`: Automatically enabled in LOSO mode to use 100% of continuous signals per recording file without intra-file truncation.

---

## 2. Automated LOSO Orchestration (`experiments/run_rnn_classifier_loso.py`)

The orchestrator script automatically discovers all subjects in the dataset, trains all $N$ folds sequentially in isolated subprocesses (to avoid TensorFlow VRAM accumulation), and generates a consolidated statistical summary.

### Configuration Template (`configs/loso_rnn_classifier/seedvig_loso_v0_cp2.yaml`)
```yaml
dataset_type: "seed_vig"
channel: "CP2"
lead_time_sec: 0.0
stride_sec: 5.0
rnn_model_path: "outputs/models/best_rnn_predictor_seedvig_CP2.h5"
learning_rate: 0.001
epochs: 10
batch_size: 32
class_weight_mode: "balanced"
save_plot: true
```

### Executing LOSO Cross-Validation
* **Run full LOSO across all discovered subjects (e.g. all 23 subjects in SEED-VIG)**:
  ```bash
  python3 experiments/run_rnn_classifier_loso.py \
      --params_file configs/loso_rnn_classifier/seedvig_loso_v0_cp2.yaml \
      --output_folder rnn_classifier_loso_seedvig
  ```

* **Run LOSO on a specific subset of subjects (e.g. Subjects 1, 2, and 3)**:
  ```bash
  python3 experiments/run_rnn_classifier_loso.py \
      --params_file configs/loso_rnn_classifier/seedvig_loso_v0_cp2.yaml \
      --output_folder rnn_classifier_loso_subset \
      --subjects 1 2 3
  ```

* **Force CPU execution**:
  ```bash
  python3 experiments/run_rnn_classifier_loso.py \
      --params_file configs/loso_rnn_classifier/seedvig_loso_v0_cp2.yaml \
      --output_folder rnn_classifier_loso_cpu \
      --force-cpu
  ```

---

## 3. LOSO Multi-Panel Evolution Overview Plot

Upon completing all folds, the orchestrator generates a comprehensive multi-panel overview plot (`loso_evolution_overview.png`) containing:

1. **Panel A (Multi-Fold Training Dynamics)**:
   - **A1. Loss Dynamics (BCE):** Overlapping training and validation loss trajectories for all $N$ folds (semi-transparent lines) with highlighted bold lines for the Global Train Mean ($\mu_{\text{treino}}$) and Global Val Mean ($\mu_{\text{validação}}$).
   - **A2. Accuracy Dynamics:** Overlapping training and validation accuracy trajectories (in $\%$) across epochs for all $N$ folds with highlighted bold mean curves.
   - **Integer Epochs:** The x-axis uses strictly discrete integer ticks ($1, 2, \dots, N$).
2. **Panel B (Subject-by-Subject Validation Performance)**:
   - Bar chart showing validation accuracy for each individual subject ($1, 2, \dots, N$).
   - Horizontal red dashed line for Global Mean Accuracy ($\mu$).
   - Shaded horizontal band for Standard Deviation ($\mu \pm \sigma$).
   - Direct percentage annotations atop each bar to identify physiological outliers.

---

## 4. LOSO Outputs and Summary

Inside the specified `--output_folder` (under `outputs/`):
* **`loso_summary.json`**: Global aggregated statistics:
  - `mean_val_accuracy` and `std_val_accuracy` ($\mu \pm \sigma$)
  - `mean_val_loss` and `std_val_loss`
  - Per-subject breakdown of validation metrics and sample counts.
* **`loso_results.jsonl`**: Detailed line-by-line log for each fold.
* **`loso_evolution_overview.png`**: The complete multi-panel visual summary.
* **`fold_subj_XX/`**: Per-fold directories containing model weights (`.h5`), training curves (`history.png`), and raw fold metrics (`metrics.json`).
* **`log.txt`**: Consolidated execution logs.

---

## 5. Running LOSO Unit Tests

To run the unit tests verifying LOSO subject discovery and multi-panel plot generation:
```bash
python3 -m unittest tests/test_rnn_loso.py
```