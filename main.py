from load_digits import load_images, load_labels
from kmeans import train_kmeans
from evaluate import test_kmeans
from visualize_centroids import save_centroids


def main():
    # Load MNIST data files
    train_images = load_images("train-images-idx3-ubyte/train-images-idx3-ubyte")
    train_labels = load_labels("train-labels-idx1-ubyte/train-labels-idx1-ubyte")
    test_images = load_images("t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
    test_labels = load_labels("t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")

    print(f"Training samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")

    # Assignment setting
    # Use 10,000 training samples for reasonable runtime.
    # The default feature is the 14x14 downsampled 196-D vector.
    K = 10
    max_iter = 20
    num_train_samples = 10000
    feature_type = "downsampled"   # You may change this to "raw" for comparison.

    # Train k-means and obtain cluster-to-label mapping.
    centroids, cluster_to_label = train_kmeans(
        train_images,
        train_labels,
        K=K,
        max_iter=max_iter,
        num_train_samples=num_train_samples,
        feature_type=feature_type,
        random_seed=0,
    )

    # Visualize the learned cluster centers.
    save_centroids(centroids, feature_type=feature_type, filename="centroids.png")

    # Evaluate on the test set.
    error_rate = test_kmeans(test_images, test_labels, centroids, cluster_to_label, feature_type=feature_type)

    print(f"Final Test Error Rate: {error_rate:.2f}%")


if __name__ == "__main__":
    main()
