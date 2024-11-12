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


def find_target_neighbors(
        data,
        data_std,
        cost_column="avg_cost_per_marker",
        county_column="County",
        target_county="Van Buren",
        n_neighbors=5
    ):
    """
    Finds the nearest neighbors of a specified target county using standardized data and extracts the original cost per marker.
    Includes the target county in the returned DataFrame.

    Parameters:
    - data (pd.DataFrame): Original data containing cost per marker and other columns.
    - data_std (pd.DataFrame): Standardized data used to find neighbors.
    - cost_column (str): Column representing the cost per marker.
    - county_column (str): Column representing the county names.
    - target_county (str): The county to find neighbors for.
    - n_neighbors (int): Number of nearest neighbors to find.

    Returns:
    - pd.DataFrame: Data of nearest neighbors with avg_cost_per_marker values, including the target county.
    """
    # Locate the index of the target county in the standardized data
    target_index = data_std.index[data[county_column] == target_county][0]

    # Initialize Nearest Neighbors model on standardized data
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto').fit(data_std.drop(columns=[county_column]))
    distances, indices = nbrs.kneighbors(data_std.drop(columns=[county_column]).iloc[[target_index]])

    # Use indices to get original data for neighbors, excluding the target county itself (first index)
    neighbors_data = data.iloc[indices[0][1:]][[county_column, cost_column]]

    # Add target county’s row to the neighbors data
    target_data = pd.DataFrame({county_column: [target_county], cost_column: [data.loc[target_index, cost_column]]})
    neighbors_data = pd.concat([target_data, neighbors_data], ignore_index=True)

    return neighbors_data

