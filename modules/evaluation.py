# modules/evaluation.py
from sklearn.metrics import silhouette_score

def evaluate_clusters(data, cluster_labels, metric="silhouette"):
    """
    Evaluates clustering performance using the specified metric.

    Parameters:
    - data (pd.DataFrame): The standardized data used for clustering.
    - cluster_labels (pd.Series or np.array): Labels from the clustering step.
    - metric (str): The evaluation metric to use, default is "silhouette".

    Returns:
    - float: The evaluation score.
    """
    if metric == "silhouette":
        try:
            score = silhouette_score(data, cluster_labels)
            print(f"Silhouette Score: {score:.4f}")
            return score
        except Exception as e:
            print(f"An error occurred while calculating the silhouette score: {e}")
            return None
    else:
        print(f"Error: Unknown metric '{metric}'")
        return None
