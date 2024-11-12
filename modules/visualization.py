# modules/visualization.py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

def visualize_clusters(data, cluster_labels, diag_kind="kde", alpha=0.6, marker_size=50, title="Cluster Visualization", palette="bright"):
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
        palette=palette,
        plot_kws={"alpha": alpha, "s": marker_size}
    )
    pair_plot.fig.suptitle(title, y=1.02)  # Adjust title position
    plt.show()

def visualize_clusters_interactive(data, cluster_labels, name_field="County", highlight_county="Van Buren",color_scale="Bluered"):
    """
    Creates an interactive scatter matrix with tooltips showing the name field for each data point.

    Parameters:
    - data (pd.DataFrame): The standardized data used for clustering.
    - cluster_labels (pd.Series or np.array): Labels from the clustering step.
    - name_field (str): The name of the column containing the identifier to display on hover.
    - highlight_county (str): The name of the county to highlight.

    Returns:
    - None: Displays the interactive plot.
    """
    # Ensure the data includes the necessary columns
    data = data.copy()
    data["Cluster"] = cluster_labels
    
    # Add a column to indicate if the point is the highlighted county
    data["Highlight"] = data["County"].apply(lambda x: x == highlight_county)

    # Select only the numeric columns for dimensions (excluding 'Cluster' and 'name_field')
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    
    # Create the interactive scatter matrix
    fig = px.scatter_matrix(
        data,
        dimensions=numeric_columns,
        color="Cluster",
        hover_name="County",  # Displays the name on hover
        title="Interactive Cluster Visualization",
        color_continuous_scale=getattr(px.colors.sequential, color_scale),
        labels={col: col.replace('_', ' ').title() for col in numeric_columns},  # Format axis labels
        symbol="Highlight",  # Different symbols for highlighted point
        symbol_map={True: "star", False: "circle"}  # Star for the highlighted county
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))  # Customize marker size and opacity

    # Update trace specifically for highlighted county to make it more visible
    fig.update_traces(selector=dict(marker_symbol="star"), marker=dict(size=12, color="red"))

    # # Add annotation for the highlighted county in each relevant subplot
    # if highlight_county in data["County"].values:
    #     highlight_data = data[data["County"] == highlight_county]
    #     for i, row in highlight_data.iterrows():
    #         for x_dim in numeric_columns:
    #             for y_dim in numeric_columns:
    #                 if x_dim != y_dim:  # Only place annotations in off-diagonal plots
    #                     fig.add_annotation(
    #                         x=row[x_dim],
    #                         y=row[y_dim],
    #                         text=highlight_county,
    #                         showarrow=True,
    #                         arrowhead=1,
    #                         ax=20,
    #                         ay=-20,
    #                         font=dict(color="red", size=10),
    #                         xref=f"x{x_dim}",
    #                         yref=f"y{y_dim}",
    #                     )

    fig.show()
