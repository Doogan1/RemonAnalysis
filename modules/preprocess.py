# modules/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def standardize(data, include_columns=None, method="z-score"):
    """
    Standardizes only numeric features in the dataset using the specified method.
    
    Parameters:
    - data (pd.DataFrame): The input data to standardize.
    - method (str): The standardization method ("z-score" or "min-max").

    Returns:
    - pd.DataFrame: The standardized data with only numeric columns.
    """
    if data is None:
        print("Error: No data provided for standardization.")
        return None
    
    # Select only specified columns if provided
    if include_columns:
        data = data[include_columns]

    # Selecting only numeric columns
    numeric_data = data.select_dtypes(include=["float64", "int64"])
    
    if method == "z-score":
        scaler = StandardScaler()
    elif method == "min-max":
        scaler = MinMaxScaler()
    else:
        print(f"Error: Unknown standardization method '{method}'")
        return None
    
    # Standardizing the numeric columns
    try:
        numeric_data = pd.DataFrame(scaler.fit_transform(numeric_data), columns=numeric_data.columns)
        print(f"Data standardized using {method} scaling.")
        return numeric_data  # Return only standardized numeric columns
    except Exception as e:
        print(f"An error occurred during standardization: {e}")
        return None
