# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load your data (assuming you have it in a DataFrame)
# Replace 'your_data.csv' with the actual path to your Excel file
data = pd.read_csv('~/Desktop/4-2/SOP-ECO/DERC3.csv')

# Extract relevant features (e.g., DerivativeChange)
X = data[['2014','2015','2016','2017','2018','2019','2020','2021','2022']]

# Determine the optimal number of clusters (K)
# You can use techniques like the Elbow Method or Silhouette Score
# For demonstration, let's try a range of K values (2 to 10)
distortions = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)

# Plot the Elbow Method to find the optimal K
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), distortions, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Distortion (within-cluster sum of squares)')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()

# Based on the plot, choose an appropriate K (e.g., where the distortion starts to level off)

# Perform k-means clustering with the chosen K
optimal_k = 5  # Example: You can adjust this based on the Elbow Method
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Now 'data' contains an additional column 'Cluster' with cluster assignments
# You can explore the clusters further or use them for analysis

# Note: Remember to preprocess your data (scaling, handling missing values, etc.) before clustering.
