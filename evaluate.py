import numpy as np
from features import extract_feature


def test_kmeans(test_images, test_labels, centroids, cluster_to_label, feature_type="downsampled"):
    """
    Evaluate k-means clustering as a digit recognizer using the cluster-to-label mapping.

    Args:
        test_images: list of 28x28 images
        test_labels: list of integer labels in {0, ..., 9}
        centroids: numpy array of shape (K, D)
        cluster_to_label: numpy array of shape (K,)
        feature_type: "downsampled" for 196-D feature or "raw" for 784-D feature

    Returns:
        error_rate: percentage of misclassified samples
    """
    num_errors = 0
    num_samples = len(test_images)

    for image, true_label in zip(test_images, test_labels):
        # TODO 1: Extract the feature vector of the test image.
        x = extract_feature(image, feature_type=feature_type)

        # TODO 2: Compute the squared Euclidean distance from x to each centroid.
        distances = np.zeros(centroids.shape[0])
        for k in range(centroids.shape[0]):
            distances[k] = np.sum((x - centroids[k]) ** 2)

        # TODO 3: Find the nearest cluster and convert it to a predicted digit label.
        nearest_cluster = np.argmin(distances)
        predicted_label = cluster_to_label[nearest_cluster]

        # TODO 4: Count misclassified samples.
        if predicted_label != true_label:
            num_errors += 1

    # TODO 5: Compute and return the error rate (%).
    error_rate = (num_errors / num_samples) * 100
    print(f"Test Error Rate: {error_rate:.2f}%")
    return error_rate
