# modules/clustering.py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import time

def find_optimal_clusters(data, max_clusters=10):
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o', linestyle='-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()

    n_clusters = None
    while n_clusters is None:
        try:
            n_clusters = int(input("After closing the elbow plot, enter the number of clusters to proceed: "))
            if 1 <= n_clusters <= max_clusters:
                plt.close()  # Close plot and proceed
            else:
                print(f"Please enter a number between 1 and {max_clusters}.")
                n_clusters = None  # Reset if invalid
        except ValueError:
            print("Invalid input. Please enter an integer.")
    return n_clusters

def perform_clustering(data, n_clusters):
    """
    Performs KMeans clustering on the data with the specified number of clusters.

    Parameters:
    - data (pd.DataFrame): The standardized data.
    - n_clusters (int): The number of clusters to create.

    Returns:
    - pd.Series: Cluster labels for each data point.
    """
    print(f"Running KMeans clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    try:
        cluster_labels = kmeans.fit_predict(data)
        print(f"Clustering completed with {n_clusters} clusters.")
        return cluster_labels
    except Exception as e:
        print(f"An error occurred during clustering: {e}")
        return None
