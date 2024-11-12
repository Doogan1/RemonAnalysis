# modules/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def standardize(data, method="z-score"):
    """
    Standardizes features in the dataset using the specified method.
    
    Parameters:
    - data (pd.DataFrame): The input data to standardize.
    - method (str): The standardization method ("z-score" or "min-max").

    Returns:
    - pd.DataFrame: The standardized data.
    """
    if data is None:
        print("Error: No data provided for standardization.")
        return None
    
    # Selecting the columns with numeric data types
    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
    
    if method == "z-score":
        scaler = StandardScaler()
    elif method == "min-max":
        scaler = MinMaxScaler()
    else:
        print(f"Error: Unknown standardization method '{method}'")
        return None
    
    # Standardizing the numeric columns
    try:
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        print(f"Data standardized using {method} scaling.")
        return data
    except Exception as e:
        print(f"An error occurred during standardization: {e}")
        return None
