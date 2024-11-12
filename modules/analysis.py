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
from sklearn.neighbors import NearestNeighbors
import pandas as pd

def find_target_neighbors(data, data_std, cost_column="avg_cost_per_marker", county_column="County", target_county="Van Buren", n_neighbors=5, metrics=["wetlandd", "pop_d", "roadoverarea"]):
    """
    Finds the nearest neighbors of a specified target county using standardized data and specified metrics, and extracts the original cost per marker.
    Includes the target county in the returned DataFrame.

    Parameters:
    - data (pd.DataFrame): Original data containing cost per marker and other columns.
    - data_std (pd.DataFrame): Standardized data used to find neighbors.
    - cost_column (str): Column representing the cost per marker.
    - county_column (str): Column representing the county names.
    - target_county (str): The county to find neighbors for.
    - n_neighbors (int): Number of nearest neighbors to find.
    - metrics (list): List of metric columns to use for k-NN.

    Returns:
    - pd.DataFrame: Data of nearest neighbors with avg_cost_per_marker values, including the target county.
    """
    # Ensure the county names are included in data_std
    if county_column not in data_std.columns:
        data_std[county_column] = data[county_column].values

    # Filter data_std to only include specified metrics and county names
    data_for_knn = data_std[[county_column] + metrics].copy()

    # Find the index of the target county
    target_index = data_for_knn.index[data_for_knn[county_column] == target_county][0]

    # Prepare data for NearestNeighbors (exclude county column)
    knn_data = data_for_knn.set_index(county_column)[metrics]

    # Initialize Nearest Neighbors model on the selected metrics
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto').fit(knn_data)
    distances, indices = nbrs.kneighbors([knn_data.loc[target_county]])

    # Get the indices of the nearest neighbors (excluding the target county itself)
    neighbor_indices = indices[0][1:]

    # Map indices back to county names
    neighbor_counties = knn_data.iloc[neighbor_indices].index.tolist()

    # Extract neighbors' avg_cost_per_marker and county names from original data
    neighbors_data = data[data[county_column].isin(neighbor_counties)][[county_column, cost_column]]

    # Add target county’s data to the neighbors data
    target_data = data[data[county_column] == target_county][[county_column, cost_column]]
    neighbors_data = pd.concat([target_data, neighbors_data], ignore_index=True)

    return neighbors_data



