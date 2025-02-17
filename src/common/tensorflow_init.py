# src/common/tensorflow_init.py
# src/common/tensorflow_init.py
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

try:
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    tf.compat.v1.disable_eager_execution()
    tf.config.set_visible_devices([], 'GPU')
except ImportError:
    pass

logger = logging.getLogger(__name__)
logger.info(
    "Note: To enable AVX2 and FMA instructions, TensorFlow needs to be rebuilt "
    "with appropriate compiler flags. This warning can be safely ignored if "
    "you're primarily using PyTorch."
)