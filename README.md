# Remon Analysis

This project analyzes county data and provides visualizations for clustering, k-nearest neighbors, and other statistical insights.

## Configuration File (`config.json`)

The configuration file contains the settings used to control the pipeline. Below is an explanation of each parameter:

### Top-Level Keys
- **`file_path`**: The path to the input CSV file containing county data. Example: `"data/pop_wetland_road_by_county.csv"`
- **`analysis_type`**: Determines which analysis to run. Options:
  - `"knn"`: Runs the k-nearest neighbors analysis.
  - `"clustering"`: Runs clustering analysis.

---

### KNN Settings (`knn`)
- **`target_county`**: The county for which nearest neighbors are calculated. Example: `"Van Buren"`
- **`n_neighbors`**: The number of neighbors to find. Example: `5`
- **`metrics`**: List of column names used to calculate similarity. Example: `["wetlandd", "pop_d", "roadoverarea"]`
- **`verbose`**: Toggles debug information for KNN analysis. Options:
  - `true`: Print debug statements.
  - `false`: Suppress debug output.

---

### Clustering Settings (`clustering`)
- **`max_clusters`**: Maximum number of clusters to test for the elbow method. Example: `10`

---

### Standardization Settings (`standardization`)
- **`method`**: Determines the method for data standardization. Options:
  - `"z-score"`: Standardizes data to have mean 0 and standard deviation 1.
  - `"min-max"`: Scales data to the range [0, 1].

---

### Filtering Settings (`filter`)
- **`exclude_counties`**: List of county names to exclude from the analysis. Example:
  ```json
  "filter": {
    "exclude_counties": ["Bay", "Kent", "Livingston"]
  }
  ```

---

### Visualization Settings (`visualization`)
- **`show_labels`**: Determines how to display labels for scatter plots. Options:
  - `"on_hover"`: Show county names when hovering over points.
  - `"next_to_points"`: Show county names as text next to scatter points.

---

## Example `config.json`
```json
{
    "file_path": "data/pop_wetland_road_by_county.csv",
    "analysis_type": "knn",
    "knn": {
        "target_county": "Van Buren",
        "n_neighbors": 5,
        "metrics": ["wetlandd", "pop_d", "roadoverarea"],
        "verbose": true
    },
    "clustering": {
        "max_clusters": 10
    },
    "standardization": {
        "method": "z-score"
    },
    "filter": {
        "exclude_counties": ["Bay", "Kent", "Livingston"]
    },
    "visualization": {
        "show_labels": "on_hover"
    }
}
```
