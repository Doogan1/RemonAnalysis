# modules/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def standardize(df, method="z-score"):
    if method == "z-score":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        print(f"Unknown standardization method: {method}")
        return None

    scaled_data = scaler.fit_transform(df)
    # Create a DataFrame with the same index and columns as the input
    standardized_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
    print(f"Data standardized using {method} scaling.")
    return standardized_df



def filter_data(data, config):
    """
    Filters data based on criteria specified in the config file.

    Parameters:
    - data (pd.DataFrame): The input DataFrame to filter.
    - config (dict): Configuration dictionary with filtering criteria.

    Returns:
    - pd.DataFrame: The filtered DataFrame.
    """
    # Filter based on excluded counties, if specified
    exclude_counties = config.get("filters", {}).get("exclude_counties", [])
    if exclude_counties:
        if "County" in data.columns:
            data = data[~data["County"].isin(exclude_counties)]
            print(f"Excluding counties: {exclude_counties}. Rows remaining after filter: {len(data)}")
        else:
            print("Warning: 'County' column not found in data.")
    return data

