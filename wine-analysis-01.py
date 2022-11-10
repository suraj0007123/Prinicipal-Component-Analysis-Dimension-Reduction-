## Importing All Necessary Libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from scipy.spatial.distance import cdist 
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

### Importing data for PCA and hierarchical clustering and Kmeans analysis
wine = pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\dimension reducation\Datasets_PCA\wine.csv")

wine.describe() ## To know stats about the data

wine.info() ## To know information about the data

###### Dropping uneccessary column #####
wine1 = wine.drop(["Type"], axis = 1)

# Considering only numerical data 
wine1.data = wine1.iloc[:, 0:]

# Normalizing the numerical data 
wine1_normal = scale(wine1.data)
wine1_normal
pca = PCA(n_components = 3)
pca_values = pca.fit_transform(wine1_normal)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

# PCA weights
pca.components_
pca.components_[0]

# Cumulative variance 
var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")

# PCA scores
pca_values
pca_data = pd.DataFrame(pca_values)
pca_data.columns = "PC1", "PC2", "PC3"
pca_data

plt.figure(figsize=(12,8))
sns.scatterplot(data=pca_data)
plt.show()

# for creating dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
from sklearn.cluster import AgglomerativeClustering

z = linkage(pca_data, method = "single", metric = "euclidean")
# Dendrogram on single linkage method
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram on WINE Data');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()

z = linkage(pca_data, method = "average", metric = "euclidean")
# Dendrogram on average linkage method
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram on WINE Data');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()

z = linkage(pca_data, method = "complete", metric = "euclidean")
# Dendrogram on complete linkage method
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram on WINE Data');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(pca_data) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

wine1['clusters'] = cluster_labels # creating a new column and assigning it to new column 

wine1
## Hierarchical Clustering on cluster "0"
wine1[wine1['clusters']==0]
## Hierarchical Clustering on cluster "1"
wine1[wine1['clusters']==1]
## Hierarchical Clustering on cluster "2"
wine1[wine1['clusters']==2]
## Hierarchical Clustering on cluster "3"
wine1[wine1['clusters']==3]


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 
import warnings
warnings.filterwarnings("ignore")

###### scree plot or elbow curve ############
TWSS=[]

k=list(range(1,10))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(pca_data)
    TWSS.append(kmeans.inertia_)
TWSS

# Scree plot 
plt.figure(figsize=(10,5));plt.title('Elbow Method on Wine Data');plt.plot(k, TWSS, 'ms-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

plt.figure(figsize=(12,7))
plt.scatter(pca_data['PC1'],pca_data['PC2'],pca_data['PC3'],cmap = plt.cm.rainbow)
plt.show()

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(pca_data)

model.labels_ # getting the labels of clusters assigned to each row 
clusters = pd.Series(model.labels_)  # converting numpy array into pandas series object 
pca_data['cluster'] = clusters # creating a  new column and assigning it to new column 
pca_data

pca_data.columns # To know columns names after assigning the cluster column

plt.figure(figsize=(12,8))
plt.scatter(pca_data["PC1"], pca_data["PC2"], pca_data["PC3"],c=pca_data['cluster'], cmap = plt.cm.coolwarm)
plt.show()

## Kmeans clustering on clusters "0"
pca_data[pca_data['cluster']==0]

## Kmeans clustering on clusters "1"
pca_data[pca_data['cluster']==1]

## Kmeans clustering on clusters "2"
pca_data[pca_data['cluster']==2]

## Kmeans clustering on clusters "3"
pca_data[pca_data['cluster']==3]

wine1_normal


#### Checking whether we have obtained same number of clusters with the original data.
norm_df=pd.DataFrame(wine1_normal)

norm_df

### Checking with original data using Hierarchical clustering 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

z = linkage(norm_df, method = "single", metric = "euclidean")
# Dendrogram on single linkage method
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram on WINE Data');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()

z = linkage(norm_df, method = "average", metric = "euclidean")
# Dendrogram on average linkage method 
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram on WINE Data');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()

z = linkage(norm_df, method = "complete", metric = "euclidean")
# Dendrogram on complete linkage method
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram on WINE Data');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(norm_df) 
h_complete.labels_

clusters = pd.Series(h_complete.labels_)

wine1['cluster_original_data'] = clusters

wine1

## Hierarchical clustering with cluster "0"
wine1[wine1["cluster_original_data"]==0]

## Hierarchical clustering with cluster "1"
wine1[wine1["cluster_original_data"]==1]

## Hierarchical clustering with cluster "2"
wine1[wine1["cluster_original_data"]==2]

## Hierarchical clustering with cluster "3"
wine1[wine1["cluster_original_data"]==3]


## Checking with original data using kmeans clustering 
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 
import warnings
warnings.filterwarnings("ignore")

###### scree plot or elbow curve ############
TWSS=[]

k=list(range(1,10))

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(pca_data)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot
plt.figure(figsize=(10,5));plt.plot(range(1,10), TWSS,"cs-");plt.title('Elbow Method on wine data');plt.xlabel('Optinum number of clusters');plt.show()
## The scatterplot on all 3 components 
plt.figure(figsize=(12,8))
plt.scatter(pca_data['PC1'],pca_data['PC2'],pca_data['PC3'], cmap=plt.cm.rainbow)
plt.show()

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(norm_df)

model.labels_ # getting the labels of clusters assigned to each row 
clusters = pd.Series(model.labels_)  # converting numpy array into pandas series object 
pca_data['cluster_original_data'] = clusters
pca_data['cluster_original_data_1'] = clusters # creating a  new column and assigning it to new column 

pca_data

plt.figure(figsize=(12,8))
plt.scatter(pca_data['PC1'],pca_data['PC2'],pca_data['PC3'], c=pca_data["cluster_original_data_1"],cmap=plt.cm.rainbow)
plt.show()
## kmeans clustering with clusters "0"
pca_data[pca_data['cluster_original_data_1']==0]

## kmeans clustering with clusters "1"
pca_data[pca_data['cluster_original_data_1']==1]

## kmeans clustering with clusters "2"
pca_data[pca_data['cluster_original_data_1']==2]

## kmeans clustering with clusters "3"
pca_data[pca_data['cluster_original_data']==3]


### Checking same number of clusters 
## Hierarchical clustering 
wine1['clusters'].value_counts()

pca_data['cluster_original_data'].value_counts()


#### Kmeans clustering 
pca_data['cluster_original_data_1'].value_counts()

pca_data['cluster'].value_counts()



#### INTERFERENCE :-

 
#-In hierarchical clustering didn't get same number of clusters with original data
#-In KMeans clustering got same number of clusters with original data





