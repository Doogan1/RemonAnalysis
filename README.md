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

### Standardization Settings (`standardization`)

- **`method`**: Determines the method for data standardization. Options:
  - `"z-score"`: Standardizes data to have mean 0 and standard deviation 1.
  - `"min-max"`: Scales data to the range [0, 1].

---

### Filtering Settings (`filters`)

- **`exclude_counties`**: List of county names to exclude from the analysis. Example:
  ```json
  "filters": {
      "exclude_counties": ["Bay", "Kent", "Livingston"]
  }
  ```

---

### Clustering Settings (`clustering`)

- **`method`**: The clustering algorithm to use. Options:
  - `"k-means"`: Uses k-means clustering.
- **`max_clusters`**: Maximum number of clusters to test for the elbow method. Example: `10`
- **`init`**: Initialization method for the k-means algorithm. Options:
  - `"k-means++"`: A smart initialization method that speeds up convergence.
  - `"random"`: Randomly initializes cluster centroids.

---

### Evaluation Settings (`evaluation`)

- **`metric`**: The evaluation metric for clustering. Options:
  - `"silhouette"`: Silhouette score, which measures how similar data points are within a cluster compared to other clusters.

---

### KNN Settings (`knn`)

- **`target_county`**: The county for which nearest neighbors are calculated. Example: `"Van Buren"`
- **`n_neighbors`**: The number of neighbors to find. Example: `10`
- **`metrics`**: List of column names used to calculate similarity. Example: `["wetlandd", "pop_d", "roadoverarea"]`
- **`verbose`**: Toggles debug information for KNN analysis. Options:
  - `true`: Print debug statements.
  - `false`: Suppress debug output.

---

### Visualization Settings (`visualization`)

- **`diag_kind`**: Specifies the type of plot for diagonal elements in pair plots. Options:
  - `"kde"`: Kernel Density Estimate.
  - `"hist"`: Histogram.
- **`alpha`**: Transparency level of points in scatter plots. Example: `0.6`
- **`marker_size`**: Size of scatter plot markers. Example: `50`
- **`title`**: Title for cluster visualizations. Example: `"Cluster Visualization"`
- **`palette`**: Color palette for plots. Example: `"bright"`
- **`highlight_county`**: County to highlight in visualizations. Example: `"Van Buren"`
- **`color_scale`**: Color scale for visualizations. Example: `"Bluered"`
- **`show_labels`**: Determines how to display labels for scatter plots. Options:
  - `"on_hover"`: Show county names when hovering over points.
  - `"next_to_points"`: Show county names as text next to scatter points.
- **`selected_groups`**: Selects which subset of groups to display in the box plot. Options:
  - `"Target"`: Show just the target point and a degenerate box plot.
  - `"Neighbors"`: Show the n_neighbors - nearest neighbors scatter and box plot.
  - `"Target + Neighbors"`: Combine the target with its neighbors and plot the scatter points along with the box plot.

---

### Include Columns (`include_columns`)

- **`include_columns`**: Specifies which columns to include for analysis. Only these columns will be standardized and used for metrics. Example:
  ```json
  "include_columns": ["wetlandd", "pop_d", "roadoverarea"]
  ```

---

## Example `config.json`

```json
{
    "file_path": "data/pop_wetland_road_by_county.csv",
    "standardization": {
        "method": "z-score"
    },
    "filters": {
        "exclude_counties": ["Bay", "Kent", "Livingston", "Mason", "Menominee", "Muskegon", "Oakland", "Ottawa"]
    },
    "analysis_type": "knn",
    "clustering": {
        "method": "k-means",
        "max_clusters": 10,
        "init": "k-means++"
    },
    "evaluation": {
        "metric": "silhouette"
    },
    "knn": {
        "target_county": "Van Buren",
        "n_neighbors": 10,
        "metrics": ["wetlandd", "pop_d", "roadoverarea"],
        "verbose": false
    },
    "visualization": {
        "diag_kind": "kde",
        "alpha": 0.6,
        "marker_size": 50,
        "title": "Cluster Visualization",
        "palette": "bright",
        "highlight_county": "Van Buren",
        "color_scale": "Bluered",
        "show_labels": "next_to_points",
        "selected_groups": ["Target","Neighbors"]
    },
    "include_columns": ["wetlandd", "pop_d", "roadoverarea"]
}
```