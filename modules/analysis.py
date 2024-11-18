# modules/analysis.py
from sklearn.neighbors import NearestNeighbors
import pandas as pd

def find_nearest_neighbors(data, cost_column="avg_cost_per_marker", n_neighbors=5):
    """
    Finds the nearest neighbors for each data point and extracts the cost per marker of these neighbors.

    Parameters:
    - data (pd.DataFrame): Standardized data used in clustering.
    - cost_column (str): Column representing the cost per marker.
    - n_neighbors (int): Number of nearest neighbors to find.

    Returns:
    - pd.DataFrame: Original data with an additional column for neighbor cost distributions.
    """
    # Initialize Nearest Neighbors model
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)

    # Collect cost per marker distributions for each data point's neighbors
    neighbor_costs = []
    for i, neighbors in enumerate(indices):
        # Exclude the point itself (first neighbor in the result) and get cost for actual neighbors
        neighbor_cost = data.iloc[neighbors[1:]][cost_column].values
        neighbor_costs.append(neighbor_cost)

    # Add the neighbor costs as a new column to the original data
    data[f"{cost_column}_neighbors"] = neighbor_costs
    return data


# modules/analysis.py

def find_target_neighbors(data, data_std,
    cost_column="avg_cost_per_marker",
    county_column="County",
    target_county="Van Buren",
    n_neighbors=5,
    metrics=["wetlandd", "pop_d", "roadoverarea"],
    verbose=False):
    """
    Finds the nearest neighbors of a specified target county using standardized data and specified metrics, and extracts the original cost per marker.
    Includes the target county in the returned DataFrame.
    """
    def log(message):
        """Helper function to print messages if verbose is enabled."""
        if verbose:
            print(message)

    # Ensure the county names are included in data_std
    if county_column not in data_std.columns:
        data_std[county_column] = data[county_column].values
    log(f"Step 1: County column '{county_column}' confirmed in data_std.")

    # Filter data_std to only include specified metrics and county names
    try:
        data_for_knn = data_std[[county_column] + metrics].copy()
        log(f"Step 2: Data prepared for KNN with columns: {data_for_knn.columns}")
    except KeyError as e:
        log(f"Error: Missing columns in data_std - {e}")
        return None

    # Find the index of the target county
    try:
        target_index = data_for_knn.index[data_for_knn[county_column] == target_county][0]
        log(f"Step 3: Target index for '{target_county}' found at {target_index}")
    except IndexError as e:
        log(f"Error: Target county '{target_county}' not found in data_for_knn.")
        log(f"Available counties in data_for_knn: {data_for_knn[county_column].unique()}")
        return None

    # Prepare data for NearestNeighbors (exclude county column)
    knn_data = data_for_knn.set_index(county_column)[metrics]
    log(f"Step 4: Prepared knn_data with shape {knn_data.shape} for metrics: {metrics}")

    # Initialize Nearest Neighbors model on the selected metrics
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto').fit(knn_data)
    log("Step 5: NearestNeighbors model initialized and fitted.")

    try:
        distances, indices = nbrs.kneighbors(knn_data.loc[[target_county]])
        log(f"Step 6: Nearest neighbors found. Indices: {indices}")
    except KeyError as e:
        log(f"Error: Target county '{target_county}' not found in knn_data. Available counties: {knn_data.index.tolist()}")
        return None

    # Get the indices of the nearest neighbors (excluding the target county itself)
    neighbor_indices = indices[0][1:]
    log(f"Step 7: Neighbor indices (excluding target): {neighbor_indices}")

    # Map indices back to county names
    neighbor_counties = knn_data.iloc[neighbor_indices].index.tolist()
    log(f"Step 8: Neighbor counties identified: {neighbor_counties}")

    # Extract neighbors' avg_cost_per_marker and county names from original data
    neighbors_data = data[data[county_column].isin(neighbor_counties)][[county_column, cost_column]]
    log("Step 9: Neighbors' data extracted with cost values.")
    log(f"Step 9b: Neighbor counties in knn_data: {neighbor_counties}")
    log(f"Step 9c: Neighbor counties in data: {data[county_column].unique()}")
    log(f"step 9ci: Length of neighbor counties in data: {len(data[county_column].unique())}")
    log(f"Step 9d: Filtered neighbors_data:\n{neighbors_data}")

    # Add target county’s data to the neighbors data
    target_data = data[data[county_column] == target_county][[county_column, cost_column]]
    neighbors_data = pd.concat([target_data, neighbors_data], ignore_index=True)
    print(neighbors_data)
    log("Step 10: Target county data added to neighbors data.")

    return neighbors_data




