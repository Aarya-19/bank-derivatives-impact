import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load your data from the Excel file
data = pd.read_csv('~/Desktop/4-2/SOP-ECO/DERC2.csv')
# file_path = r"~/Desktop/4-2/SOP-ECO/DERC.xlsx"
# data = pd.read_excel(file_path)
df = pd.DataFrame(data, columns=["BankName","2014","2015","2016","2017","2018","2019","2020","2021","2022"])
print(df)
# Assuming you have a DataFrame 'df' with columns 'BankName' and 'DerivativesChange'

# Extract the relevant feature (derivatives year-on-year change)
# Extract the relevant features (derivatives year-on-year change)
X = df[['2014','2015','2016','2017','2018','2019','2020','2021','2022']].values.reshape(-1, 1)


# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize an empty list to store silhouette scores
silhouette_scores = []

# Try different cluster numbers (e.g., from 2 to 10)
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plot the silhouette scores
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.grid(True)
plt.show()
