import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

#Load data
data = pd.read_csv("Data.csv")

#print(data.columns)

data = data.rename(columns={
    'StateProgressReport2021_County' : 'County',
    'StateProgressReport2021_Total_Remon_corners_in_County' : 'NumCorners',
    'StateProgressReport2021_Aveverage_Spent_per_Corner_Completed' : 'AvgCostPerCorner',
    'StateProgressReport2021_TotalArea' : 'Area',
    'StateProgressReport2021_TotalAreaRUCA7Plus' : 'AreaRUCA7Plus',
    'StateProgressReport2021_TotalAreaRUCA8Plus' : 'AreaRUCA8Plus',
    'StateProgressReport2021_TotalAreaRUCA9Plus' : 'AreaRUCA9Plus',
    'StateProgressReport2021_PercentRUCA7Plus' : 'PercentRUCA7Plus',
    'StateProgressReport2021_PercentRUCA8Plus' : 'PercentRUCA8Plus',
    'StateProgressReport2021_PercentRUCA9Plus' : 'PercentRUCA9Plus',
    'StateProgressReport2021_RoadsPerSqMile' : 'RoadDensity',
    'StateProgressReport2021_RoadCount' : 'RoadCount',
    'StateProgressReport2021_AverageTRI' : 'AvgTRI',
    'StateProgressReport2021_PopulationDensity' : 'PopulationDensity',
    'StateProgressReport2021_WeightedTRI' : 'WeightedTRI'
})


#Select features for clustering
selected_features = ['PopulationDensity', 'RoadDensity', 'wetland_area']
features = data[selected_features]

#Standardize features
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

# # Elbow method for determining optimal number of clusters
# wcss = []
# for k in range(1,11):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(standardized_features)
#     wcss.append(kmeans.inertia_)

# # Plot WCSS to find the elbow

# plt.plot(range(1, 11), wcss, marker='o')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS')
# plt.title('Elbow Method for Optimal K')
# plt.show()

#Analysis reveals optimal_k=4
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(standardized_features)
labels = kmeans.labels_

# sil_score = silhouette_score(standardized_features, labels)
# print("Silhouette Score: ", sil_score)

data['Cluster'] = labels

data.to_csv('Data_with_clusters.csv', index=False)


#View cluster assignment
# print(data[['County', 'Cluster']])

#Calculate cluster means to understand the characteristics of each cluster
# cluster_means = data.groupby('Cluster').mean(numeric_only=True)
# print(cluster_means)

#Visualize the clusters
# plt.scatter(data['PopulationDensity'], data['AvgTRI'], c=data['Cluster'], cmap='viridis')
# plt.xlabel('Population Density')
# plt.ylabel('Terrain Ruggedness')
# plt.title('Cluster Visualization')
# plt.colorbar(label='Cluster')
# plt.show()

# selected_features.append('Cluster')
# sns.pairplot(data[selected_features], hue='Cluster', palette='viridis', markers=['o', 's', 'D'], diag_kind='kde')

# plt.show()

# # Visualize to try to understand clusters
# selected_features.append('AvgCostPerCorner')

# for feature in selected_features:
#     plt.figure(figsize=(8,6))
#     sns.boxplot(x='Cluster', y=feature, data=data)
#     plt.title(f'Distribution of {feature} by Cluster')
#     plt.show()

# Calculate average cost per corner for each cluster

# cost_by_cluster = data.groupby('Cluster')['AvgCostPerCorner'].mean()
# print(cost_by_cluster)


# plt.figure(figsize=(12,16))

# sns.boxplot(x='Cluster', y='AvgCostPerCorner', data=data, palette='viridis')

# for cluster in data['Cluster'].unique():
#     cluster_data = data[data['Cluster'] == cluster]

#     for i in range(len(cluster_data)):
#         plt.text(
#             x=cluster_data['Cluster'].values[i],
#             y=cluster_data['AvgCostPerCorner'].values[i],
#             s=cluster_data['County'].values[i],
#             fontdict=dict(color='red', size=8),
#             ha='center', va='bottom'
#         )

# plt.title('Distribution of Average Cost per Corner by Cluster')
# plt.xlabel('Cluster')
# plt.ylabel('Average Cost per Corner')
# plt.show()

# cluster_data = data[data['Cluster'] == 3]
# plt.figure(figsize=(8,50))
# sns.boxplot(y='AvgCostPerCorner', data=cluster_data, color='skyblue')

# for i in range(len(cluster_data)):
#     plt.text(
#         x=0,
#         y=cluster_data['AvgCostPerCorner'].values[i],
#         s=cluster_data['County'].values[i],
#         fontdict=dict(color='red', size=8),
#         ha='right'
#     )

# plt.title('Cost per Corner for Cluster 3')
# plt.ylabel('Average Cost per Corner')
# plt.xlabel('')

# plt.show()