# modules/clustering.py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

def find_optimal_clusters(data, max_clusters=10):
    """
    Determines the optimal number of clusters using the elbow method.

    Parameters:
    - data (pd.DataFrame): The standardized data.
    - max_clusters (int): The maximum number of clusters to try.

    Returns:
    - None: Displays a plot for the elbow method.
    """
    inertias = []
    # Calculate inertia for each number of clusters
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    # Plot the elbow graph
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o', linestyle='-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()

    # Prompt the user for the number of clusters to use
    while True:
        try:
            n_clusters = int(input("Enter the optimal number of clusters based on the plot: "))
            if 1 <= n_clusters <= max_clusters:
                return n_clusters
            else:
                print(f"Please enter a number between 1 and {max_clusters}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

def perform_clustering(data, n_clusters):
    """
    Performs KMeans clustering on the data with the specified number of clusters.

    Parameters:
    - data (pd.DataFrame): The standardized data.
    - n_clusters (int): The number of clusters to create.

    Returns:
    - pd.Series: Cluster labels for each data point.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    print(f"Clustering completed with {n_clusters} clusters.")
    return cluster_labels
