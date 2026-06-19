import argparse
import scalogram_cnn_project.settings.config as config
from scalogram_cnn_project.rpca_preprocessing.process_rpca_simple import process_single_image

############################################################################
## PARSER CONFIGURATION
############################################################################

parser = argparse.ArgumentParser(description="Apply RPCA Simple to a Single Image")
parser.add_argument("--image_path", type=str, default="data/generated_scalograms_ALL_gray_overlap0.733_extra_input_example/img_0a85796bce.png", help="Path to the test image")
parser.add_argument("--output_folder", type=str, default="rpca_simple_output", help="Output folder name inside OUTPUT_DIR")
parser.add_argument("--lambdas", type=float, nargs="+", default=[0.05, 0.10, 0.125, 0.15, 0.175, 0.2, 0.25, 0.30], help="RPCA lambdas to test")
parser.add_argument("--mu", type=float, default=None, help="RPCA mu parameter (None for default)")
parser.add_argument("--tolerance", type=float, default=None, help="RPCA tolerance parameter (None for default)")
parser.add_argument("--max_iteration", type=int, default=None, help="RPCA max iteration parameter (None for default)")
parser.add_argument("--cmap", type=str, default="gray", help="Colormap to use")

args = parser.parse_args()

############################################################################
## RUN PARAMETERS
############################################################################

IMAGE_PATH =  args.image_path
OUTPUT_FOLDER = config.OUTPUT_DIR / args.output_folder
LAMBDAS_TO_TEST = args.lambdas
MU = args.mu
TOLERANCE = args.tolerance
MAX_ITERATION = args.max_iteration
CMAP = args.cmap

if __name__ == "__main__":
    for lamb in LAMBDAS_TO_TEST:
        process_single_image(
            IMAGE_PATH, OUTPUT_FOLDER, lamb=lamb, mu=MU, tolerance=TOLERANCE, max_iteration=MAX_ITERATION, cmap=CMAP
        )