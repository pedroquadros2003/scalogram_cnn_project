import scalogram_cnn_project.settings.config as config
from scalogram_cnn_project.rpca_preprocessing.process_rpca_simple import process_single_image

if __name__ == "__main__":
    # Direct path to the test image
    IMAGE_PATH = config.DATA_DIR / "generated_scalograms_ALL_gray_overlap0.733_extra_input_example" / "img_0a85796bce.png"
    
    OUTPUT_FOLDER = config.OUTPUT_DIR / "rpca_simple_output"

    # RPCA Parameters to test (None will use the default mathematical optimum)
    LAMBDAS_TO_TEST = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    MU = None
    TOLERANCE = None
    MAX_ITERATION = None
    CMAP = "gray"

    for lamb in LAMBDAS_TO_TEST:
        process_single_image(
            IMAGE_PATH, OUTPUT_FOLDER, lamb=lamb, mu=MU, tolerance=TOLERANCE, max_iteration=MAX_ITERATION, cmap=CMAP
        )