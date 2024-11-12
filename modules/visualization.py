# modules/visualization.py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def visualize_clusters(data, cluster_labels, diag_kind="kde", alpha=0.6, marker_size=50, title="Cluster Visualization"):
    """
    Creates a pair plot of the data with clusters distinguished by color.

    Parameters:
    - data (pd.DataFrame): The standardized data used for clustering.
    - cluster_labels (pd.Series or np.array): Labels from the clustering step.
    - diag_kind (str): Type of plot on the diagonal ("kde" or "hist").
    - alpha (float): Transparency of the plot markers.
    - marker_size (int): Size of the plot markers.
    - title (str): Title for the plot.

    Returns:
    - None: Displays the plot.
    """
    # Convert data to DataFrame if necessary and add cluster labels
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    data["Cluster"] = cluster_labels

    # Create the pair plot with KDE or histogram on the diagonal
    pair_plot = sns.pairplot(
        data,
        hue="Cluster",
        diag_kind=diag_kind,
        plot_kws={"alpha": alpha, "s": marker_size}
    )
    pair_plot.fig.suptitle(title, y=1.02)  # Adjust title position
    plt.show()
