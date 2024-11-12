# modules/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def standardize(data, method="z-score"):
    """
    Standardizes only numeric features in the dataset using the specified method.
    
    Parameters:
    - data (pd.DataFrame): The input data to standardize.
    - method (str): The standardization method ("z-score" or "min-max").

    Returns:
    - pd.DataFrame: The standardized data.
    """
    if data is None:
        print("Error: No data provided for standardization.")
        return None
    
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
        data[numeric_data.columns] = scaler.fit_transform(numeric_data)
        print(f"Data standardized using {method} scaling.")
        return data
    except Exception as e:
        print(f"An error occurred during standardization: {e}")
        return None
