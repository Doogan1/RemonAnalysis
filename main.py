import json
from modules.load_data import load_csv
from modules.preprocess import standardize
from modules.clustering import find_optimal_clusters, perform_clustering
from modules.evaluation import evaluate_clusters
from modules.analysis import find_target_neighbors, find_nearest_neighbors
from modules.visualization import (
    visualize_clusters_interactive,
    visualize_cost_distribution,
    visualize_target_neighbor_distribution,
    visualize_neighbor_cost_distribution,
    visualize_3d_neighbors,
    visualize_3d_with_costs
)

# Load configuration file
with open("config.json", "r") as f:
    config = json.load(f)

def run_knn_analysis(data, data_std, config):
    # Get parameters from config
    target_county = config["knn"]["target_county"]
    n_neighbors = config["knn"]["n_neighbors"]
    metrics = config["knn"]["metrics"]

    # Find the target county's nearest neighbors
    neighbors_data = find_target_neighbors(
        data, data_std,
        cost_column="avg_cost_per_marker",
        county_column="County",
        target_county=target_county,
        n_neighbors=n_neighbors,
        metrics=metrics
    )

    # Extract neighbor county names for visualization (excluding the target county)
    neighbor_names = neighbors_data[neighbors_data["County"] != target_county]["County"].tolist()

    # Visualize 3D metrics space using standardized data
    visualize_3d_neighbors(
        data_std, 
        target_county=target_county, 
        neighbors=neighbor_names, 
        metrics=metrics,
        county_column="County"
    )

    visualize_3d_with_costs(data_std, metrics=["wetlandd", "pop_d", "roadoverarea"])

    # Visualize the distribution for the target county's nearest neighbors
    visualize_target_neighbor_distribution(
        neighbors_data, 
        cost_column="avg_cost_per_marker", 
        county_column="County",
        target_county=target_county
    )


def run_clustering_analysis(data_std, config):
    # Exclude non-numeric columns and any columns we don't want for clustering
    clustering_data = data_std.drop(columns=["County", "avg_cost_per_marker"], errors="ignore")

    # Determine optimal clusters
    n_clusters = find_optimal_clusters(clustering_data, max_clusters=config["clustering"]["max_clusters"])

    # Perform clustering
    cluster_labels = perform_clustering(clustering_data, n_clusters)
    data_std["Cluster"] = cluster_labels

    # Visualize clusters with interactivity
    visualize_clusters_interactive(
        data_std, 
        cluster_labels,
        name_field="County",
        highlight_county=config["visualization"].get("highlight_county", "Van Buren"),
        color_scale=config["visualization"].get("color_scale", "Bluered")
    )

    # Visualize cost distribution for each cluster
    visualize_cost_distribution(data_std, cluster_labels, cost_column="avg_cost_per_marker", name_column="County")


def run_pipeline(config):
    # Load data
    data = load_csv(config["file_path"])
    if data is None:
        print("Exiting pipeline due to data loading error.")
        return

    # Extract columns for reference and remove them from data for standardization
    county_names = data["NAME"]
    cost_per_corner = data["Aveverage Spent per Corner Completed"]

    # Preprocess: Standardize only the clustering features
    clustering_features = data.drop(columns=["NAME", "Aveverage Spent per Corner Completed"])
    data_std = standardize(clustering_features, method=config["standardization"]["method"])
    if data_std is None:
        print("Exiting pipeline due to preprocessing error.")
        return

    # Append County and avg_cost_per_marker columns for analyses that need them
    data_std["avg_cost_per_marker"] = cost_per_corner
    data_std["County"] = county_names
    data["County"] = county_names
    data["avg_cost_per_marker"] = cost_per_corner

    # Run the selected analysis based on config
    analysis_type = config.get("analysis_type", "knn")
    if analysis_type == "knn":
        run_knn_analysis(data, data_std, config)
    elif analysis_type == "clustering":
        run_clustering_analysis(data_std, config)
    else:
        print(f"Unknown analysis type: {analysis_type}")

if __name__ == "__main__":
    run_pipeline(config)
