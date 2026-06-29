import numpy as np
from features import extract_feature


def train_kmeans(train_images, train_labels, K=10, max_iter=20, num_train_samples=10000,
                 feature_type="downsampled", random_seed=0):
    """
    Train k-means clustering on MNIST images and assign each cluster to a digit label.

    Important:
        K-means itself must not use labels during clustering.
        The labels are used only after clustering, to interpret each cluster by majority voting.

    Args:
        train_images: list of 28x28 images
        train_labels: list of integer labels in {0, ..., 9}
        K: number of clusters
        max_iter: number of k-means iterations
        num_train_samples: number of training samples to use
        feature_type: "downsampled" for 196-D feature or "raw" for 784-D feature
        random_seed: random seed for reproducible initialization

    Returns:
        centroids: numpy array of shape (K, D)
        cluster_to_label: numpy array of shape (K,), where cluster_to_label[k]
                          is the digit label assigned to cluster k
    """
    rng = np.random.default_rng(random_seed)

    # Use only a subset of training data for reasonable runtime.
    images = train_images[:num_train_samples]
    labels = np.array(train_labels[:num_train_samples])

    # TODO 1: Extract feature vectors for all selected training images.
    # X should have shape (N, D), where D is 196 for "downsampled" or 784 for "raw".
    X = None
    for img in images:
        feature_vector = extract_feature(img, feature_type=feature_type)
        # Append feature_vector to X (you may initialize X as an empty list and convert to numpy array later).
        if X is None:
            X = np.array([feature_vector])
        else:
            X = np.vstack([X, feature_vector])      

    # TODO 2: Initialize K centroids by randomly selecting K samples from X.
        
    random_indices = rng.choice(X.shape[0], size=K, replace=False)  
    centroids = X[random_indices]
    

    for it in range(max_iter):
        # TODO 3: Compute the squared Euclidean distance from each sample to each centroid.
        # distances should have shape (N, K).
        distances = np.zeros((X.shape[0], K))
        for i in range(X.shape[0]):
            for k in range(K):
                distances[i, k] = np.sum((X[i] - centroids[k]) ** 2)        

        # TODO 4: Assign each sample to the nearest centroid.
        assignments = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            assignments[i] = np.argmin(distances[i])

        # TODO 5: Update each centroid as the mean of samples assigned to it.
        # If a cluster has no assigned samples, reinitialize it using a random sample from X.
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            assigned_samples = X[assignments == k]
            if len(assigned_samples) > 0:
                new_centroids[k] = assigned_samples.mean(axis=0)
            else:
                # Reinitialize empty cluster with a random sample from X
                new_centroids[k] = X[rng.choice(X.shape[0])]    

        # TODO 6: Compute and print the k-means objective value.
        # The objective is the sum of squared distances from samples to their assigned centroids.
        objective = 0.0
        for i in range(X.shape[0]):
            objective += np.sum((X[i] - centroids[assignments[i]]) ** 2)
        print(f"Iteration {it + 1}/{max_iter} - objective: {objective:.4f}")

        centroids = new_centroids

    # TODO 7: After the final iteration, assign each training sample to the nearest centroid again.
    final_assignments = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        final_assignments[i] = np.argmin([np.sum((X[i] - centroids[k]) ** 2) for k in range(K)])    

    # TODO 8: Build the cluster-to-label mapping by majority voting.
    # For each cluster k, find the most frequent digit label among samples assigned to k.
    cluster_to_label = np.zeros(K, dtype=int)
    for k in range(K):
        assigned_labels = labels[final_assignments == k]
        cluster_to_label[k] = np.bincount(assigned_labels).argmax()

    print("Training completed.")
    print("Cluster-to-label mapping:")
    for k in range(K):
        print(f"Cluster {k} -> Digit {cluster_to_label[k]}")

    return centroids, cluster_to_label
