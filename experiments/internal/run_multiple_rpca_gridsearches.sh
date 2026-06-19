#!/bin/bash

# Determine project root directory (one level above experiments/ folder)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change directory to project root to ensure all relative paths resolve correctly
cd "$PROJECT_ROOT" || exit 1

# General configuration parameters
INPUT_DATASET="data/generated_scalograms_ALL_gray_overlap0.733_extra_input_example"
PARAMS_FILE="parameter_searches/test_isolated.yaml"
MODEL_VER="v1"
RUNNER_VER="v1"

# Lambdas from 0.14 to 0.18 with step 0.005
LAMBDAS=("0.140" "0.145" "0.150" "0.155" "0.160" "0.165" "0.170" "0.175" "0.180")

for lamb in "${LAMBDAS[@]}"
do
    echo "========================================================================="
    echo "Processing RPCA Isolated for lambda=$lamb"
    echo "========================================================================="

    # Run RPCA Isolated for the current lambda value
    ./run_until_it_ends.sh experiments/apply_rpca_isolated.py \
        --input_folder "$INPUT_DATASET" \
        --output_folder "isolated_result_lambda_$lamb" \
        --cmap "gray" \
        --lamb "$lamb"

    # Define paths of the generated L (Low-Rank) and S (Sparse) component datasets
    L_FOLDER="outputs/isolated_result_lambda_$lamb/L"
    S_FOLDER="outputs/isolated_result_lambda_$lamb/S"

    echo "-------------------------------------------------------------------------"
    echo "Running Grid Search for L (Low-Rank) component | lambda=$lamb"
    echo "-------------------------------------------------------------------------"
    ./run_until_it_ends.sh experiments/run_gridsearch.py \
        --input_folder="$L_FOLDER" \
        --output_folder="gridsearch_isolated_lambda_${lamb}_L" \
        --model="$MODEL_VER" \
        --model_runner="$RUNNER_VER" \
        --params_file="$PARAMS_FILE"

    echo "-------------------------------------------------------------------------"
    echo "Running Grid Search for S (Sparse) component | lambda=$lamb"
    echo "-------------------------------------------------------------------------"
    ./run_until_it_ends.sh experiments/run_gridsearch.py \
        --input_folder="$S_FOLDER" \
        --output_folder="gridsearch_isolated_lambda_${lamb}_S" \
        --model="$MODEL_VER" \
        --model_runner="$RUNNER_VER" \
        --params_file="$PARAMS_FILE"
done
echo "========================================================================="
echo "All experiments finished successfully! Summary of results:"
echo "========================================================================="

for lamb in "${LAMBDAS[@]}"
do
    L_FILE="outputs/gridsearch_isolated_lambda_${lamb}_L/progress.json"
    S_FILE="outputs/gridsearch_isolated_lambda_${lamb}_S/progress.json"

    if [ -f "$L_FILE" ]; then
        ACC_L=$(venv_wsl/bin/python -c "import json; val=json.load(open('$L_FILE')).get('model_00000'); print('%.2f' % val if isinstance(val, (int, float)) else 'N/A')" 2>/dev/null)
    else
        ACC_L="N/A"
    fi

    if [ -f "$S_FILE" ]; then
        ACC_S=$(venv_wsl/bin/python -c "import json; val=json.load(open('$S_FILE')).get('model_00000'); print('%.2f' % val if isinstance(val, (int, float)) else 'N/A')" 2>/dev/null)
    else
        ACC_S="N/A"
    fi

    echo "Lambda: $lamb | Low-Rank (L) Acc: ${ACC_L}% | Sparse (S) Acc: ${ACC_S}%"
done
echo "========================================================================="
