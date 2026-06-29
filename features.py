import numpy as np


def image_to_vector(image, normalize=True):
    """
    Convert a 28x28 image into a 784-dimensional raster-scan vector.
    """
    x = np.array(image).flatten().astype(float)
    if normalize:
        x = x / 255.0
    return x


def image_to_downsampled_vector(image, normalize=True):
    """
    Convert a 28x28 image into a 14x14 downsampled vector (196 dimensions).

    The downsampling is done by averaging each non-overlapping 2x2 block.
    This feature is used as the default representation for the k-means assignment.
    """
    img = np.array(image).astype(float)
    img_small = img.reshape(14, 2, 14, 2).mean(axis=(1, 3))
    x = img_small.flatten()
    if normalize:
        x = x / 255.0
    return x


def extract_feature(image, feature_type="downsampled"):
    """
    Extract an image feature vector.

    Args:
        image: 28x28 grayscale image
        feature_type: "downsampled" for 196-D feature or "raw" for 784-D feature

    Returns:
        x: normalized feature vector
    """
    if feature_type == "downsampled":
        return image_to_downsampled_vector(image)
    elif feature_type == "raw":
        return image_to_vector(image)
    else:
        raise ValueError("feature_type must be either 'downsampled' or 'raw'")
