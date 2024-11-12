import json
from modules.load_data import load_csv
from modules.preprocess import standardize
from modules.clustering import find_optimal_clusters, perform_clustering
from modules.evaluation import evaluate_clusters
from modules.visualization import visualize_clusters, visualize_clusters_interactive

# Load configuration file
with open("config.json", "r") as f:
    config = json.load(f)

def run_pipeline(config):
    # Load data
    data = load_csv(config["file_path"])
    if data is None:
        print("Exiting pipeline due to data loading error.")
        return

    # Keep a reference to the county name 'NAME'
    county_names = data["NAME"]

    # Preprocess data
    data_std = standardize(data, include_columns=config.get("include_columns"),  method=config["standardization"]["method"])
    if data_std is None:
        print("Exiting pipeline due to preprocessing error.")
        return

    # Determine optimal clusters and prompt user input
    n_clusters = find_optimal_clusters(data_std, max_clusters=config["clustering"]["max_clusters"])

    # Perform clustering
    cluster_labels = perform_clustering(data_std, n_clusters)

    # Evaluate clustering
    score = evaluate_clusters(data_std, cluster_labels, metric=config["evaluation"]["metric"])
    if score is None:
        print("Exiting pipeline due to evaluation error.")
        return

    data_std["County"] = county_names

    # Visualize clusters with Plotly for interactivity
    visualize_clusters_interactive(
        data_std, cluster_labels, name_field="County",
        highlight_county=config["visualization"].get("highlight_county", "Van Buren"),
        color_scale=config["visualization"].get("color_scale","Bluered")
    )


    # # Visualize clusters with parameters from config
    # visualize_clusters(
    #     data_std,
    #     cluster_labels,
    #     diag_kind=config["visualization"].get("diag_kind", "kde"),
    #     alpha=config["visualization"].get("alpha", 0.6),
    #     marker_size=config["visualization"].get("marker_size", 50),
    #     title=config["visualization"].get("title", "Cluster Visualization")
    # )

if __name__ == "__main__":
    run_pipeline(config)
