import tensorflow as tf

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("scalogram_cnn_project").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info("Built with CUDA:", tf.test.is_built_with_cuda())
logger.info("GPUs:", tf.config.list_physical_devices('GPU'))