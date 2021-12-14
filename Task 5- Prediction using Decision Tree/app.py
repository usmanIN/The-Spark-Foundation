import pandas as pd
import matplotlib.pyplot as plt

#Load the Iris Dataset
df = pd.read_csv("Iris.csv",index_col="Id")
df.head()
df2 = df.iloc[:,:-1]

#Now we going to removes rist & last columns of Iris dataset i.e. Species
df = df2.values

#Now we're going to apply custering algorithms to identify the species clusters
from sklearn.cluster import KMeans

range_value = [i for i in range(1,11)]
wcss =[] # within cluster sum of squere

for i in range_value:
    km = KMeans(n_clusters = i, init ='k-means++',max_iter=300, n_init=10,random_state=42)
    km.fit(df)    
    wcss.append(km.inertia_)
    

#Using the elbow method to find the optimal number of clusters
plt.plot(range_value,wcss)
plt.title("Elbow method for find optimal value of clusters")
plt.xlabel("Clusters")
plt.ylabel("Within Clusters Sum of Square")
plt.legend()

#As we see there after point -> 3 clusters doesn't decrease significantly with every iteration.

k = 3 #The optimal values of clusters is 3

km = KMeans(n_clusters = k, init ='k-means++',max_iter=300, n_init=10,random_state=42)
predict = km.fit_predict(df)

#Visualizing the Clusters
plt.scatter(df[predict==0,0],df[predict==0,1],s = 100,label="Iris-setosa")
plt.scatter(df[predict==1,0],df[predict==1,1],s = 100,label="Iris-versicolour")
plt.scatter(df[predict==2,0],df[predict==2,1],s = 100,label="Iris-virginica")
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:,1], s = 100, c="red", label = 'Centroids')
plt.title('Clusters Of Clients')
plt.legend()

#Prepare the new IRIS Dataset
df2['cluster'] = pd.Series(predict,index= df2.index)
df2

x = df2.iloc[:,:-1]
y = df2.iloc[:,-1:]


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

# defining the decision tree algorithm
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(x_train,y_train)

dpredict = dtree.predict(x_test)

from sklearn.metrics import accuracy_score

print('Accuracy: %.3f' % accuracy_score(y_test, dpredict))

from sklearn import tree
cn=["Iris-setosa","Iris-versicolour","Iris-virginica"]
plt.figure(figsize=(12,20)) 
tree.plot_tree(dtree,feature_names=x.columns,class_names=cn,filled=True)



import seaborn as sns
data = pd.read_csv("Iris.csv",index_col="Id")

plt.figure(figsize=(12,10)) 
column = data.columns[:-1]
n = round(len(column)/2)
i=1
for col in column:    
    plt.subplot(n, 2, i)
    plt.tight_layout()
    plt.title(col)
    #data[col].value_counts().plot(kind='bar',figsize=(15,10))
    sns.boxplot(x="Species",y=col,data=data)
    i+=1  
plt.subplot(n, 2, i)
data['Species'].value_counts().plot(kind="pie",legend=True)

