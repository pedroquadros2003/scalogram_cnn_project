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
The script generates the scalograms with overlap predetermined.

### v1
The script generates the scalograms with overlap predetermined
and power spectral features in .npy file.

---

## generator/generate_scalogram_simple

**Purpose**

It serves the purpose of generating just the first scalogram associated with an EDF file of the DROZY dataset.

**Versions**

### v0
The script generates just the first scalogram according to the arguments passed to the function.

---


# Models

**Description**

Models are function that create model and callback objects.

**Versions**

### v0
It is a model with fixed hyperparameters; its architecture matches the description of two-layered CNN-2D as described by A. Zayed (2025).

### v1
It is a model with variable hyperparameters; its architecture is a variation of the one proposed by A. Zayed (2025), as it allows the user to utilize one extra convolutional layers, as well as adjust the number of filters in each layer, the kernel size etc.


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

--

# Use of Logging Package

Instead of using print statements in the source code of the scalogram_cnn_project package, messages to the terminal are configured using the Logging package.