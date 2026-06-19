import argparse
from pathlib import Path
import scalogram_cnn_project.settings.config as config
from scalogram_cnn_project.rpca_preprocessing.process_rpca_juxtaposed import process_juxtaposed

############################################################################
## PARSER CONFIGURATION
############################################################################

parser = argparse.ArgumentParser(description="Apply RPCA Juxtaposed Preprocessing to a Folder")
parser.add_argument("--input_folder", type=str, default=str(config.DATA_DIR / "generated_scalograms_ALL_gray_overlap0.733_extra_input_example"), help="Full path of the input folder")
parser.add_argument("--output_folder", type=str, default="juxtaposed_scalograms", help="Output folder name inside OUTPUT_DIR")
parser.add_argument("--cmap", type=str, default="gray", help="Colormap to use")
parser.add_argument("--lamb", type=float, default=None, help="RPCA lambda parameter (None for default)")
parser.add_argument("--mu", type=float, default=None, help="RPCA mu parameter (None for default)")
parser.add_argument("--tolerance", type=float, default=None, help="RPCA tolerance parameter (None for default)")
parser.add_argument("--max_iteration", type=int, default=None, help="RPCA max iteration parameter (None for default)")

args = parser.parse_args()

############################################################################
## RUN PARAMETERS
############################################################################

INPUT_FOLDER = Path(args.input_folder)
OUTPUT_FOLDER = config.OUTPUT_DIR / args.output_folder
OUTPUT_L_FOLDER = OUTPUT_FOLDER / "L"
OUTPUT_S_FOLDER = OUTPUT_FOLDER / "S"

CMAP = args.cmap
LAMB = args.lamb
MU = args.mu
TOLERANCE = args.tolerance
MAX_ITERATION = args.max_iteration

if __name__ == "__main__":
    process_juxtaposed(
        INPUT_FOLDER, OUTPUT_L_FOLDER, OUTPUT_S_FOLDER, 
        lamb=LAMB, mu=MU, tolerance=TOLERANCE, max_iteration=MAX_ITERATION, cmap=CMAP
    )