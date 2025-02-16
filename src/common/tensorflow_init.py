# src/common/tensorflow_init.py
import os
import logging

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ERROR only

# Initialize absl logging first
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

# Try to initialize TensorFlow with proper settings
try:
    import tensorflow as tf

    # Configure TensorFlow
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Allow memory growth to avoid pre-allocating all GPU memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
      # Disable eager execution
    tf.compat.v1.disable_eager_execution()
    tf.config.set_visible_devices([], 'GPU')
except ImportError:
    # TensorFlow not installed, which is fine since we primarily use PyTorch
    pass

# Note about CPU instructions
logger = logging.getLogger(__name__)
logger.info(
    "Note: To enable AVX2 and FMA instructions, TensorFlow needs to be rebuilt "
    "with appropriate compiler flags. This warning can be safely ignored if "
    "you're primarily using PyTorch."
)