# modules/visualization.py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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

def visualize_cost_distribution(data, cluster_labels, cost_column="avg_cost_per_marker", name_column="County"):
    """
    Creates a box plot to show the distribution of average cost per marker for each cluster.

    Parameters:
    - data (pd.DataFrame): The original dataset containing the cost column and other information.
    - cluster_labels (pd.Series or np.array): Labels from the clustering step.
    - cost_column (str): The name of the column representing the cost per marker.
    - name_column (str): The column to display on hover (e.g., county name).

    Returns:
    - None: Displays the interactive plot.
    """
    # Copy data and add cluster labels for plotting
    plot_data = data.copy()
    plot_data["Cluster"] = cluster_labels

    # Create a box plot for each cluster
    fig = px.box(
        plot_data,
        x="Cluster",
        y="avg_cost_per_marker",
        points="all",  # Display all points overlaid on the box plot
        hover_name="County",  # Show the name (county) on hover
        title=f"Distribution of {cost_column.replace('_', ' ').title()} by Cluster",
        labels={"Cluster": "Cluster", cost_column: "Avg Cost per Marker"}
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.show()


def visualize_neighbor_cost_distribution(data, cost_column="avg_cost_per_marker", county_column="County"):
    """
    Creates a box plot showing the distribution of cost per marker for the nearest neighbors of each data point.

    Parameters:
    - data (pd.DataFrame): Data containing neighbor cost distributions.
    - cost_column (str): The column representing the cost per marker.

    Returns:
    - None: Displays the plot.
    """
    # Expand neighbor costs into individual rows for easy plotting
    neighbor_data = pd.DataFrame({
        "Point": data.index.repeat(len(data[f"{cost_column}_neighbors"].iloc[0])),
        f"{cost_column}_neighbors": [item for sublist in data[f"{cost_column}_neighbors"] for item in sublist],
        county_column: data[county_column].repeat(len(data[f"{cost_column}_neighbors"].iloc[0])).values
    })

    # Create box plot
    fig = px.box(
        neighbor_data,
        x="Point",
        y=f"{cost_column}_neighbors",
        hover_data={county_column: True},
        title=f"Distribution of {cost_column.replace('_', ' ').title()} for Nearest Neighbors",
        labels={"Point": "Data Point", f"{cost_column}_neighbors": f"{cost_column.replace('_', ' ').title()}"}
    )
    fig.show()



def visualize_target_neighbor_distribution(
        neighbors_data,
        cost_column="avg_cost_per_marker",
        county_column="County",
        target_county="Van Buren",
        config=None
    ):
    """
    Creates a combined box plot and scatter plot for the cost per marker of specified groups:
    - The target county.
    - The nearest neighbors.
    - The target county combined with its nearest neighbors.
    """
    config = config or {}  # Default to empty dict if None
    show_labels = config.get("visualization", {}).get("show_labels", "on_hover")
    selected_groups = config.get("visualization", {}).get("selected_groups", ["Target", "Neighbors", "Target + Neighbors"])

    # Add a column to distinguish between groups for plotting
    neighbors_data["Group"] = neighbors_data[county_column].apply(
        lambda x: "Target" if x == target_county else "Neighbors"
    )

    # Add a third group that combines the target county with its neighbors
    combined_data = neighbors_data.copy()
    combined_data["Group"] = "Target + Neighbors"

    # Concatenate original and combined data
    plot_data = pd.concat([neighbors_data, combined_data])

    # Filter plot_data based on selected groups
    plot_data = plot_data[plot_data["Group"].isin(selected_groups)]

    # Dynamically assign x positions based on selected groups
    group_positions = {group: i * 4 for i, group in enumerate(selected_groups)}

    # Precompute jittered x-values for unique rows
    unique_data = plot_data.drop_duplicates(subset=[county_column, "Group"])
    unique_data["Jittered X"] = np.random.uniform(
        -0.5, 0.5, size=unique_data.shape[0]
    )

    # Map the jittered x-values back to the full dataset
    plot_data["Jittered X"] = plot_data.merge(
        unique_data[[county_column, "Group", "Jittered X"]],
        on=[county_column, "Group"],
        how="left"
    )["Jittered X"]

    # Create the figure
    fig = go.Figure()

    def add_box_trace(group_name, color):
        """Helper function to add box traces."""
        fig.add_trace(go.Box(
            y=plot_data[plot_data["Group"] == group_name][cost_column],
            x=[group_positions[group_name]] * plot_data[plot_data["Group"] == group_name].shape[0],  # Fix x-position
            name=group_name,
            marker=dict(color=color),
            boxpoints=False,  # Disable default scatter points
            hovertext=plot_data[plot_data["Group"] == group_name][county_column] if show_labels == "on_hover" else None,
            hoverinfo="text+y" if show_labels == "on_hover" else "y",
            offsetgroup=group_positions[group_name]
        ))

    def add_scatter_trace(group_name, color, x_offset=2):
        """Helper function to add custom scatter points with labels."""
        group_data = plot_data[plot_data["Group"] == group_name]
        fig.add_trace(go.Scatter(
            x=group_data["Jittered X"] + group_positions[group_name] - x_offset,
            y=group_data[cost_column],
            mode="markers+text" if show_labels == "next_to_points" else "markers",
            text=group_data[county_column] if show_labels == "next_to_points" else None,
            textposition="middle right",
            marker=dict(color=color, size=8),
            showlegend=False  # Scatter traces do not need a separate legend entry
        ))

    # Add box and scatter traces for each selected group
    colors = {"Target": "blue", "Neighbors": "green", "Target + Neighbors": "purple"}
    for group in selected_groups:
        add_scatter_trace(group, colors[group])
        add_box_trace(group, colors[group])

    # Update layout for spacing and appearance
    fig.update_layout(
        title={
            "text": f"Cost per Corner Distribution for {target_county} and {len(neighbors_data)-1} Nearest Neighbors",
            "x": 0.5,  # Centers the title (0 is left, 1 is right)
            "xanchor": "center",  # Ensures the title is centered
            "font": {
                "size": 24,  # Adjust the font size
                "family": "Arial, sans-serif"  # Optional: Specify a font family
            }
        },
        yaxis_title="Avg Cost per Marker",
        xaxis=dict(
            title="Group",
            tickvals=[group_positions[group] for group in selected_groups],
            ticktext=selected_groups,
        ),
        showlegend=False
    )


    fig.show()




def visualize_3d_neighbors(data, target_county="Van Buren", neighbors=[], metrics=["Metric1", "Metric2", "Metric3"], county_column="County"):
    """
    Creates a 3D scatter plot of data points, highlighting the target county and its nearest neighbors.

    Parameters:
    - data (pd.DataFrame): Data containing the metrics and county names.
    - target_county (str): The name of the target county to highlight.
    - neighbors (list): List of nearest neighbor counties to highlight.
    - metrics (list): List of three metric column names to plot in 3D space.
    - county_column (str): Column name representing the county names.

    Returns:
    - None: Displays the 3D plot.
    """
    if len(metrics) != 3:
        print("Error: Please provide exactly three metrics for the 3D plot.")
        return

    # Filter data into different categories for coloring and symbolization
    target_data = data[data[county_column] == target_county]
    neighbors_data = data[data[county_column].isin(neighbors)]
    other_data = data[~data[county_column].isin([target_county] + neighbors)]

    # Create 3D scatter plot with separate traces
    fig = go.Figure()

    # Target county trace
    fig.add_trace(go.Scatter3d(
        x=target_data[metrics[0]],
        y=target_data[metrics[1]],
        z=target_data[metrics[2]],
        mode='markers',
        marker=dict(size=10, color='red', symbol="diamond"),
        name="Target County",
        text=target_data[county_column]
    ))

    # Nearest neighbors trace
    fig.add_trace(go.Scatter3d(
        x=neighbors_data[metrics[0]],
        y=neighbors_data[metrics[1]],
        z=neighbors_data[metrics[2]],
        mode='markers',
        marker=dict(size=7, color='blue', symbol="circle"),
        name="Neighbors",
        text=neighbors_data[county_column]
    ))

    # Other counties trace
    fig.add_trace(go.Scatter3d(
        x=other_data[metrics[0]],
        y=other_data[metrics[1]],
        z=other_data[metrics[2]],
        mode='markers',
        marker=dict(size=5, color='gray', symbol="cross"),
        name="Other Counties",
        text=other_data[county_column]
    ))

    # Set plot title and labels
    fig.update_layout(
        title={
            "text": f"3D Plot of Metrics Showing Nearest Neighbors of {target_county}",
            "x": 0.5,  # Centers the title (0 is left, 1 is right)
            "xanchor": "center",  # Ensures the title is centered
            "font": {
                "size": 24,  # Adjust the font size
                "family": "Arial, sans-serif"  # Optional: Specify a font family
            }
        },
        scene=dict(
            xaxis_title=metrics[0],
            yaxis_title=metrics[1],
            zaxis_title=metrics[2]
        )
    )

    fig.show()



def visualize_3d_with_costs(data, metrics=["Metric1", "Metric2", "Metric3"], county_column="County", cost_column="avg_cost_per_marker", use_log_scale=True):
    """
    Creates a 3D scatter plot of data points with sphere-like markers colored by avg cost per corner.
    Optionally, applies a logarithmic scale for color to improve contrast in power-law distributed costs.

    Parameters:
    - data (pd.DataFrame): Data containing the metrics, county names, and avg cost per corner.
    - metrics (list): List of three metric column names to plot in 3D space.
    - county_column (str): Column name representing the county names.
    - cost_column (str): Column representing the avg cost per corner for coloring.
    - use_log_scale (bool): If True, applies a logarithmic scale to the cost column for coloring.

    Returns:
    - None: Displays the 3D plot.
    """
    if len(metrics) != 3:
        print("Error: Please provide exactly three metrics for the 3D plot.")
        return

    # Apply log scale to the cost column if specified
    color_values = np.log10(data[cost_column]) if use_log_scale else data[cost_column]

    # Create 3D scatter plot
    fig = go.Figure()

    # Plot all counties with color based on the transformed cost values
    fig.add_trace(go.Scatter3d(
        x=data[metrics[0]],
        y=data[metrics[1]],
        z=data[metrics[2]],
        mode='markers',
        marker=dict(
            size=8,
            color=color_values,
            colorscale="Viridis",  # Alternative: "Plasma" for more contrast at lower values
            colorbar=dict(title="Avg Cost per Corner (log scale)" if use_log_scale else "Avg Cost per Corner")
        ),
        name="Counties",
        text=data[county_column]  # Hover text for county names
    ))

    # Set plot title and axis labels
    fig.update_layout(
        title={
            "text":"3D Plot of Metrics Colored by Avg Cost per Corner",
            "x": 0.5,  # Centers the title (0 is left, 1 is right)
            "xanchor": "center",  # Ensures the title is centered
            "font": {
                "size": 24,  # Adjust the font size
                "family": "Arial, sans-serif"  # Optional: Specify a font family
            }
        },
        scene=dict(
            xaxis_title=metrics[0],
            yaxis_title=metrics[1],
            zaxis_title=metrics[2]
        )
    )

    fig.show()
