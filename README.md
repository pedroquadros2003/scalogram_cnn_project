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

---

# Generator Scripts

## generator/generate_scalogram_batch

**Purpose**

It serves the purpose of generating all scalograms associated with an EDF file of the DROZY dataset.

**Versions**

### v0
The script generates the scalograms with overlap predetermined, but this can be cause data leakage when applying random undersampling and splitting the dataset in training and testinfg parts.

---

## generator/generate_scalogram_simple

**Purpose**

It serves the purpose of generating just the first scalogram associated with an EDF file of the DROZY dataset.

**Versions**

### v0
The script generates just the first scalogram according to the arguments passed to the function.

---

# Model Runners

## model_runner_mix

**Description**

It runs a model that does not differentiate between scalograms from different channels during training and testing, mixing them in the process.

## model_runner_separate

**Description**

It runs a model that, in order to deal with multichannel information, increases the input depth for receiving more color channels at a time.


**Versions**

### v0
It is prepared to receive scalograms from a selected set of channels, using a color map to the user's choice. It suffers from data leakage, due to the the overlap between the epochs considered.

### v1
It is also prepared to receive scalograms from a selected set of channels, using a color map to the user's choice. It solves the problem of data leakage by destinating the first seven minutes of each session to training and the rest to testing.

### v2
It is also prepared to receive scalograms from a selected set of channels, using a color map to the user's choice. It solves the problem of data leakage by applying a Leave-One-Subject-Out (LOSO) validation.

--

# Use of Logging Package

Instead of using print statements in the source code of the scalogram_cnn_project package, messages to the terminal are configured using the Logging package.