# From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually.
#Dataset : "Iris.csv"

import pandas as pd

df = pd.read_csv("Iris.csv",index_col=['Id'])
df.head()


import matplotlib.pyplot as plt
import seaborn as sns

sns.jointplot(x="SepalLengthCm", y="SepalWidthCm",data=df)
sns.jointplot(x="PetalLengthCm", y="PetalWidthCm",data=df)

df.hist(edgecolor='black', linewidth=1.2)

sns.pairplot(df, hue='Species', size=3)


# prepareing the data variables
x = df.iloc[:,:-1]
y = df.iloc[:,-1:]


xClus = df.iloc[:,:-1].values

#Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
sse = []
for k in range(1,11):
    km = KMeans(n_clusters=k,init='k-means++',max_iter=300,n_init=10,random_state=42)
    km.fit(xClus)
    sse.append(km.inertia_)

plt.plot(range(1,11),sse)   
plt.title("The Elbow Method")
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")

#Applying K-means to the Iris dataset
model = KMeans(n_clusters=k,init='k-means++',max_iter=300,n_init=10,random_state=42)
y_means = model.fit_predict(xClus)


# Visualizing the Clusters

plt.scatter(xClus[y_means == 0,0], xClus[y_means == 0,1], s = 100 , c='red', label="Iris-setosa")
plt.scatter(xClus[y_means == 1,0], xClus[y_means == 1,1], s = 100 , c='blue', label="Iris-versicolour")
plt.scatter(xClus[y_means == 2,0], xClus[y_means == 2,1], s = 100 , c='green', label="Iris-virginica")
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:,1], s = 100, c="black", label = 'Centroids')
plt.title('Clusters Of Clients')
plt.xlabel("")
plt.ylabel("")
plt.legend()


 