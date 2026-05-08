from scalogram_cnn_project.scalogram_generation_seed_vig.generator_scalogram_simple import generate_scalogram

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("scalogram_cnn_project").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    # Run the generator

    generate_scalogram(subject = 1,
                       channel="O1", 
                       epoch_index=10,
                       epoch_duration=8.0, # in seconds
                       wavelet_type = 'morl',
                       freq_min=3,
                       freq_max=30,
                       cmap="viridis",
                       ## Size of the first scalogram generated, according to A. Zayed (2025)
                       width_px = 662,  
                       height_px = 536,
                       dpi = 100,
                       show_bands = True,
                       ## Final sized of the scalogram, designed to be input of a CNN-2D
                       final_width_px = 256,
                       final_height_px = 256,
                       )

