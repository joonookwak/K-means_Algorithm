import numpy as np
import matplotlib.pyplot as plt


def save_centroids(centroids, feature_type="downsampled", filename="centroids.png"):
    """
    Save cluster centers as digit-like images.

    Args:
        centroids: numpy array of shape (K, D)
        feature_type: "downsampled" or "raw"
        filename: output image filename
    """
    K = centroids.shape[0]
    fig, axes = plt.subplots(1, K, figsize=(1.4 * K, 1.6))

    if K == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        if feature_type == "downsampled":
            img = centroids[k].reshape(14, 14)
        elif feature_type == "raw":
            img = centroids[k].reshape(28, 28)
        else:
            raise ValueError("feature_type must be either 'downsampled' or 'raw'")

        ax.imshow(img, cmap="gray")
        ax.set_title(f"C{k}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved centroid visualization to {filename}")
