import json
import pandas as pd
from modules.load_data import load_csv
from modules.preprocess import (
    standardize,
    filter_data
)
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
    target_county = config["knn"]["target_county"]

    # Check for NaN values in County column
    if data_std["County"].isnull().any():
        print("Warning: NaN values detected in 'County' column of data_std.")
    
    # Ensure target county exists in both data and data_std
    if target_county in data["County"].values:
        print(f"'{target_county}' found in raw data (data).")
    else:
        print(f"Error: '{target_county}' not found in raw data (data).")
    
    if target_county in data_std["County"].values:
        print(f"'{target_county}' found in standardized data (data_std).")
    else:
        print(f"Error: '{target_county}' not found in standardized data (data_std).")

    # Check if data_std contains NaN values in the metrics columns
    metrics = config["knn"].get("metrics", ["wetlandd", "pop_d", "roadoverarea"])
    if data_std[metrics].isnull().values.any():
        print("Warning: NaN values detected in metrics columns in data_std.")

    # Proceed with finding neighbors if target county is present
    neighbors_data = find_target_neighbors(
        data, data_std,
        cost_column="avg_cost_per_marker",
        county_column="County",
        target_county=target_county,
        n_neighbors=config["knn"]["n_neighbors"],
        metrics=metrics,
        verbose=config["knn"].get("verbose", False)
    )
    
    if neighbors_data is None:
        print("Error: Neighbors data was not returned. Exiting.")
        return

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
        target_county=target_county,
        config=config
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

def find_neighbors_for_all(data, data_std, config, output_file="data/raw/county_neighbors.csv"):
    """
    Find the nearest neighbors for all counties in the dataset and save to a CSV.
    """
    results = []

    # Iterate over all counties in the dataset
    for county in data_std["County"].unique():
        print(f"Processing neighbors for: {county}")
        neighbors_data = find_target_neighbors(
            data=data,
            data_std=data_std,
            cost_column="avg_cost_per_marker",
            county_column="County",
            target_county=county,
            n_neighbors=config["knn"]["n_neighbors"],
            metrics=config["knn"]["metrics"],
            verbose=config["knn"].get("verbose", False)
        )

        if neighbors_data is not None:
            # Add source county to the results
            neighbors_data["SourceCounty"] = county
            results.append(neighbors_data)

    # Combine all results into a single DataFrame
    all_neighbors = pd.concat(results, ignore_index=True)

    # Save to CSV
    all_neighbors.to_csv(output_file, index=False)
    print(f"All counties' nearest neighbors saved to {output_file}")

    return all_neighbors


def run_pipeline(config):
    # Load data
    data = load_csv(config["file_path"])
    if data is None:
        print("Exiting pipeline due to data loading error.")
        return

    # Apply filtering based on config
    data = filter_data(data, config)
    
    # Verify that the target county is still in the data after filtering
    target_county = config["knn"]["target_county"]
    if target_county not in data["NAME"].values:
        print(f"Error: Target county '{target_county}' was filtered out. Check filter criteria.")
        return  # Exit early if target county is missing

    # Extract columns for reference
    county_names = data["NAME"]
    cost_per_corner = data["Aveverage Spent per Corner Completed"]

    # Preprocess: Standardize only the clustering features
    features_to_standardize = data.drop(columns=["NAME", "Aveverage Spent per Corner Completed"])
    data_std = standardize(features_to_standardize, method=config["standardization"]["method"])
    if data_std is None:
        print("Exiting pipeline due to preprocessing error.")
        return

    # Add the reference columns back to the standardized data
    data_std["avg_cost_per_marker"] = cost_per_corner
    data_std["County"] = county_names

    # Verify that no NaN values are in the 'County' column after assignment
    if data_std["County"].isnull().any():
        print("Error: NaN values detected in 'County' column of data_std after assignment.")
        return  # Exit if data is invalid

    # Keep original data columns intact in 'data' for any raw-data analysis
    data["County"] = county_names
    data["avg_cost_per_marker"] = cost_per_corner

    # Run the selected analysis based on config
    analysis_type = config.get("analysis_type", "knn")
    if analysis_type == "knn":
        run_knn_analysis(data, data_std, config)
    elif analysis_type == "clustering":
        run_clustering_analysis(data_std, config)
    elif analysis_type == "all_neighbors":
        find_neighbors_for_all(data, data_std, config)
    else:
        print(f"Unknown analysis type: {analysis_type}")




if __name__ == "__main__":
    run_pipeline(config)
