import scalogram_cnn_project.settings.config as config
from scalogram_cnn_project.rpca_preprocessing.process_rpca_juxtaposed import process_juxtaposed

if __name__ == "__main__":
    # Replace with your target folder name
    FOLDER_NAME = "useless_viridis"
    CMAP = "viridis"

    INPUT_FOLDER = config.OUTPUT_DIR / FOLDER_NAME
    OUTPUT_L_FOLDER = config.OUTPUT_DIR / FOLDER_NAME / "juxtaposed_scalograms_L"
    OUTPUT_S_FOLDER = config.OUTPUT_DIR / FOLDER_NAME / "juxtaposed_scalograms_S"

    # RPCA Parameters (use None for default values)
    LAMB = None
    MU = None
    TOLERANCE = None
    MAX_ITERATION = None

    process_juxtaposed(INPUT_FOLDER, OUTPUT_L_FOLDER, OUTPUT_S_FOLDER, lamb=LAMB, mu=MU, tolerance=TOLERANCE, max_iteration=MAX_ITERATION, cmap=CMAP)