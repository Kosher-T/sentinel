# This module defines the parameters for the VFI model being monitored.
# The idea is that if you switch to a different model (e.g., VGG, Custom CNN),
# you only change this file and the embedding generation code.

EMBEDDING_MODEL_TYPE = "MobileNetV2"
EMBEDDING_INPUT_SHAPE = (224, 224, 3) # Required input shape for the model
EMBEDDING_FEATURE_COUNT = 1280 # Output feature vector length (MobileNetV2's final dense layer size)