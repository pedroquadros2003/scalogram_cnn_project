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

This module provides a pipeline to forecast physiological/EEG signals into subsequent future minutes (signal reconstruction) using Recurrent Neural Networks (RNNs) in Keras/TensorFlow. The pipeline is designed to load signals in a database-agnostic manner, train forecasting models using chronological splits to avoid data leakage, run inference, and plot comparisons.


---

## 1. Project Organization

### Data Abstraction (`utils/`)
* **`src/scalogram_cnn_project/utils/signal_data.py`**: Contains `SignalData`, a unified class storing 2D raw signal time-series, channel names, and sampling frequency. It provides helper methods to extract individual channels and slice specific time windows (in minutes).
* **`src/scalogram_cnn_project/utils/signal_loader.py`**: Contains `SignalLoader`, exposing static methods to load SEED-VIG (`.mat` structs) and DROZY (`.edf` via MNE) signal files and parse them into standardized `SignalData` objects.
* **`src/scalogram_cnn_project/utils/plot_results.py`**: Formats and draws comparison plots showing the input (past) signal, the predicted future signal, and the actual ground truth signal (if available in the file).

### Recurrent Model Architectures (`models_for_prediction/`)
* **`model_predict_v0.py`**: Implements an **LSTM Encoder-Decoder (Seq2Seq)** architecture for sequence-to-sequence time series prediction.
* **`model_predict_v1.py`**: Implements a **GRU Direct Projection** architecture, mapping a sequence to future samples using a dense projection layer.
* **`model_predict_builder.py`**: Factory function to dynamically instantiate and compile prediction models based on a version code (e.g. `v0`, `v1`) and a parameters dictionary.

---

## 2. Parameter Configurations (`configs/model_training/`)

Configuration templates for prediction models are placed under the `/configs/model_training` directory.
Example: `/configs/model_training/seedvig_predict_example.yaml`
```yaml
dataset_type: "seed_vig"
channel: "O1"
subjects: [1, 2, 3]       # Subject IDs to filter training files
model_version: "v0"
input_min: 5.0            # Input window duration in minutes
predict_min: 2.0          # Future prediction duration in minutes
stride_sec: 30.0          # Stride for sliding windows
epochs: 10
batch_size: 32
latent_dim: 64
learning_rate: 0.001
train_split: 0.8          # Chronological train/validation fraction
resample_freq: null       # Optional downsampling frequency in Hz (e.g. 20.0 to speed up)
force_cpu: false          # Set to true to force CPU execution and avoid GPU OOMs
output_model: null
```

---

## 3. Execution and Usage Examples

### A. Training the RNN Forecaster

Run `experiments/train_rnn.py` to train a model. The script supports loading parameters from a YAML file via `--config`, filtering subjects via `--subjects`, and setting the chronological train-test split via `--train-split`.

#### Data Leakage Prevention (Overlap Gap Rejection)
To prevent temporal data leakage caused by overlapping sliding windows, the script performs a chronological split *per file*. It calculates the overlapping transition gap:
$$\text{neglected\_windows} = \lceil \frac{T_{in} + T_{out}}{\text{stride}} \rceil$$
It trains on the first portion of windows, discards the transition windows, and validates on the remaining subsequent windows.

#### Downsampling and Memory Optimization
EEG signals typically have high sampling rates (e.g. 200 Hz). Training RNNs directly on raw signals for several minutes results in extremely long sequence lengths (e.g. 60,000 steps for a 5-minute input window), causing GPU memory (VRAM) exhaustion (OOM) or slow training.
- **`--resample-freq`**: Downsamples the EEG signal to the specified frequency (in Hz) during loading. Setting this to a value like `20.0` or `10.0` Hz dramatically shortens sequence lengths, speeding up training by 10x-20x.
- **`--force-cpu`**: Forces TensorFlow to run training on the system CPU instead of the GPU. This is recommended to avoid VRAM crashes on devices with limited GPU memory.

* **Training via YAML config**:
  ```bash
  python3 experiments/train_rnn.py --config configs/model_training/seedvig_predict_example.yaml
  ```

* **Training via CLI overrides**:
  ```bash
  python3 experiments/train_rnn.py --dataset-type seed_vig --channel O1 --model-version v0 --subjects 1 2 3 --epochs 15 --train-split 0.75 --resample-freq 20.0 --force-cpu
  ```

### B. Running Predictions (run_pipeline)

Run `experiments/run_pipeline.py` to load a signal, extract an input window, perform the RNN forecast, and save the outputs to the outputs folder.

#### Automatic File Resolution
You do not need to write absolute paths. If the specified `--file` does not exist directly, the script will automatically check inside the respective raw dataset directory (`SEED_VIG_DIR` for SEED-VIG, or `DROZY_DIR/psg` for DROZY).

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

1. **A reconstructed future signal** saved to a MATLAB `.mat` file (e.g. `outputs/predicted_10_20151125_noon_O1.mat`). The MAT file contains a dictionary with the following keys:
   * `predicted_signal`: 1D array of the predicted future signal amplitude.
   * `sfreq`: Sampling frequency of the signal.
   * `channel`: The predicted channel name.
   * `start_min` / `end_min`: Input window boundaries in minutes.
   * `predict_min`: Forecasted duration in minutes.
2. **A comparison plot** comparing the input, predicted signal, and actual ground truth saved to `outputs/plot_10_20151125_noon_O1.png`.

*Note: The default directory for these files is `outputs/` (configured dynamically via config.OUTPUT_DIR), but you can specify a custom output folder by passing the `--output-dir` argument (e.g. `--output-dir path/to/custom_folder/`).*

# RNN-MLP Sleepiness Classification Pipeline

**Description**

This module implements a Two-Stage coupled architecture where a Multi-Layer Perceptron (MLP) binary classifier is stacked directly on top of a frozen pre-trained RNN forecaster model. The coupled model classifies whether the subject is alert or drowsy based on the temporal signal windows.

Training is performed on standard-scaled inputs using Binary Crossentropy loss, and evaluated with **Accuracy** as the final metric.

---

## 1. Classification Configuration (`configs/model_training_rnn_classifier/`)

Create configuration files under `configs/model_training_rnn_classifier/`:

* **SEED-VIG Configuration** (e.g. `configs/model_training_rnn_classifier/seedvig_classify_example.yaml`):
  ```yaml
  dataset_type: "seed_vig"
  channel: "CP2"
  subjects: [1, 2, 3]
  input_min: 5.0
  predict_min: 2.0
  stride_sec: 30.0
  rnn_model_path: "outputs/models/rnn_predict_v0_seed_vig_CP2.h5"  # Pre-trained RNN forecaster model
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
  train_split: 0.8
  output_model: null
  ```

* **DROZY Configuration** (e.g. `configs/model_training_rnn_classifier/drozy_classify_example.yaml`):
  ```yaml
  dataset_type: "drozy"
  channel: "C3"
  subjects: [1, 2, 3]
  input_min: 5.0
  predict_min: 2.0
  stride_sec: 30.0
  rnn_model_path: "outputs/models/rnn_predict_v0_drozy_C3.h5"  # Pre-trained RNN forecaster model
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
  train_split: 0.8
  drowsiness_threshold: 4  # KSS threshold for DROZY dataset
  output_model: null
  ```

---

## 2. Executing Training

Run `experiments/train_rnn_classifier.py` to couple the pre-trained RNN and train the MLP classification layers.

* **Training via YAML config**:
  ```bash
  python3 experiments/train_rnn_classifier.py --config configs/model_training_rnn_classifier/seedvig_classify_example.yaml
  ```

* **Training with CLI overrides and CPU execution**:
  ```bash
  python3 experiments/train_rnn_classifier.py \
      --dataset-type seed_vig \
      --channel CP2 \
      --rnn-model-path outputs/models/rnn_predict_v0_seed_vig_CP2.h5 \
      --epochs 10 \
      --force-cpu
  ```

---

## 3. Running Integration Tests

To run the integration tests verifying the classification pipeline functionality:
```bash
python3 -m unittest tests/test_rnn_classification.py
```

# RNN Hyperparameter Grid Search

**Description**

This module provides wrappers (`run_rnn_gridsearch.py` and `run_rnn_classifier_gridsearch.py`) to perform grid search over hyperparameter configurations matching the format used in the CNN pipeline (`mode: fixed/choice/...` and `value: ...`).

Each candidate combination runs inside an isolated subprocess to prevent VRAM memory leaks or GPU out-of-memory errors in TensorFlow, communicating final validation metrics via a temporary JSON file.

---

## 1. Configurations (`configs/hyperparameter_search_rnn/`)

Parameter spaces are defined under `configs/hyperparameter_search_rnn/`:

* **RNN Forecasting Search** (e.g. `configs/hyperparameter_search_rnn/forecast_gridsearch_example.yaml`):
  ```yaml
  MODEL_HYPER_PARAMS:
    latent_dim:
      mode: "choice"
      values: [16, 32]
  MODEL_TRAIN_PARAMS:
    learning_rate:
      mode: "choice"
      values: [0.01, 0.001]
    epochs:
      mode: "fixed"
      values: [5]
    batch_size:
      mode: "fixed"
      values: [32]
    dataset_type:
      mode: "fixed"
      values: ["seed_vig"]
    channel:
      mode: "fixed"
      values: ["CP2"]
    resample_freq:
      mode: "fixed"
      values: [20.0]
  ```

* **RNN Coupled Classification Search** (e.g. `configs/hyperparameter_search_rnn/classifier_gridsearch_example.yaml`):
  ```yaml
  MODEL_HYPER_PARAMS:
    learning_rate:
      mode: "choice"
      values: [0.01, 0.001]
  MODEL_TRAIN_PARAMS:
    epochs:
      mode: "fixed"
      values: [5]
    batch_size:
      mode: "fixed"
      values: [32]
    dataset_type:
      mode: "fixed"
      values: ["seed_vig"]
    channel:
      mode: "fixed"
      values: ["CP2"]
    resample_freq:
      mode: "fixed"
      values: [20.0]
    rnn_model_path:
      mode: "fixed"
      values: ["outputs/models/rnn_predict_v0_seed_vig_CP2.h5"]
  ```

* **DROZY Forecasting Search** (e.g. `configs/hyperparameter_search_rnn/drozy_forecast_gridsearch_example.yaml`):
  ```yaml
  MODEL_HYPER_PARAMS:
    latent_dim:
      mode: "choice"
      values: [16, 32]
  MODEL_TRAIN_PARAMS:
    learning_rate:
      mode: "choice"
      values: [0.01, 0.001]
    epochs:
      mode: "fixed"
      values: [5]
    batch_size:
      mode: "fixed"
      values: [32]
    dataset_type:
      mode: "fixed"
      values: ["drozy"]
    channel:
      mode: "fixed"
      values: ["C3"]
    resample_freq:
      mode: "fixed"
      values: [20.0]
  ```

* **DROZY Coupled Classification Search** (e.g. `configs/hyperparameter_search_rnn/drozy_classifier_gridsearch_example.yaml`):
  ```yaml
  MODEL_HYPER_PARAMS:
    learning_rate:
      mode: "choice"
      values: [0.01, 0.001]
  MODEL_TRAIN_PARAMS:
    epochs:
      mode: "fixed"
      values: [5]
    batch_size:
      mode: "fixed"
      values: [32]
    dataset_type:
      mode: "fixed"
      values: ["drozy"]
    channel:
      mode: "fixed"
      values: ["C3"]
    resample_freq:
      mode: "fixed"
      values: [20.0]
    rnn_model_path:
      mode: "fixed"
      values: ["outputs/models/rnn_predict_v0_drozy_C3.h5"]
    drowsiness_threshold:
      mode: "fixed"
      values: [4]
  ```

---

## 2. Executing Grid Search

* **RNN Forecasting Grid Search**:
  ```bash
  python3 experiments/run_rnn_gridsearch.py \
      --output_folder rnn_forecast_search \
      --params_file configs/hyperparameter_search_rnn/forecast_gridsearch_example.yaml \
      --force-cpu
  ```

* **RNN Coupled Classification Grid Search**:
  ```bash
  python3 experiments/run_rnn_classifier_gridsearch.py \
      --output_folder rnn_classifier_search \
      --params_file configs/hyperparameter_search_rnn/classifier_gridsearch_example.yaml \
      --force-cpu
  ```

---

## 3. Running Grid Search Integration Tests

To run the integration tests verifying the grid search functionality:
```bash
python3 -m unittest tests/test_rnn_gridsearch.py
```