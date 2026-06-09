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

## Use of Logging Package

Instead of using print statements in the source code of the scalogram_cnn_project package, messages to the terminal are configured using the Logging package.

# Generator Scripts

## generator/generate_scalogram_batch

**Purpose**

It serves the purpose of generating all scalograms associated with an EDF file of the DROZY dataset.

**Versions**

### v0
The script generates the scalograms with overlap predetermined.

### v1
The script generates the scalograms with overlap predetermined
and power spectral features in .npy file.


## generator/generate_scalogram_simple

**Purpose**

It serves the purpose of generating just the first scalogram associated with an EDF file of the DROZY dataset.

**Versions**

### v0
The script generates just the first scalogram according to the arguments passed to the function.


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

These scripts apply Robust Principal Component Analysis (RPCA) to generated scalograms, separating them into a low-rank component (L) and a sparse component (S). Currently, they must be configured manually by editing the variables inside the scripts (e.g., `FOLDER_NAME`, `CMAP`, and RPCA parameters like `LAMB`) before execution.

**Scripts**

## `experiments/apply_rpca_isolated.py`
Applies RPCA to each scalogram individually in a given folder.
- **Configurable variables:** `FOLDER_NAME`, `CMAP`, `LAMB`, `MU`, `TOLERANCE`, `MAX_ITERATION`.
- **Output:** Creates two subdirectories, `isolated_scalograms_L` and `isolated_scalograms_S`, inside the target folder.

**Execution Example:**
```bash
python3 experiments/apply_rpca_isolated.py
```

## `experiments/apply_rpca_juxtaposed.py`
Applies RPCA to horizontally juxtaposed scalograms from different channels for the same epoch.
- **Configurable variables:** `FOLDER_NAME`, `CMAP`, `LAMB`, `MU`, `TOLERANCE`, `MAX_ITERATION`.
- **Output:** Creates two subdirectories, `juxtaposed_scalograms_L` and `juxtaposed_scalograms_S`, inside the target folder.

**Execution Example:**
```bash
python3 experiments/apply_rpca_juxtaposed.py
```

## `experiments/apply_rpca_simple.py`
Applies RPCA to a single image to test multiple lambda parameters. Ideal for parameter tuning and visualizing results.
- **Configurable variables:** `IMAGE_PATH`, `LAMBDAS_TO_TEST`, `CMAP`, `MU`, `TOLERANCE`, `MAX_ITERATION`.
- **Output:** Saves the decomposed L and S images for each tested lambda in the `rpca_simple_output` folder.

**Execution Example:**
```bash
python3 experiments/apply_rpca_simple.py
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

**Execution Example:**
```bash
./run_until_it_ends.sh experiments/run_gridsearch.py \
    --input_folder="generated_scalograms_ALL_gray_overlap0.733_extra_input_example" \
    --output_folder="generic_gridsearch_example" \
    --model="v1" \
    --model_runner="v1" \
    --params_file="gridsearch_example.yaml"
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
/parameter_searches
```

One can find examples of .yaml for each experiment script.