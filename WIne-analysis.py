## Importing All Necessary Libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from scipy.spatial.distance import cdist 
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

## Importing data for kmeans and PCA
wine=pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\dimension reducation\Datasets_PCA\wine.csv")

wine

wine.shape ## To know the shape of the wine data

wine.info() ## To know the info about the wine data

wine.dtypes ## To know the data-type of wine data

wine.describe() ## To know the stats about the wine data

wine.isna().sum() ## To find the null values in wine data

###### Identifying the duplicate values #######
duplicate=wine.duplicated()
duplicate
sum(duplicate)

wine.head()

###### Dropping the column from wine data which is unnecessary #####
wine1=wine.drop(["Type"],axis=1)

wine1.columns ## To know the columns names after the dropping the column

wine2=wine.values

wine2

# Normalizing the Numerical columns in wine data 
wine2_normal=scale(wine)

wine2_normal

wine3=pd.DataFrame(wine2_normal)

wine3

#### Model Building / PCA Implementing 
## Applying PCA fit transform to dataset 

pca=PCA(n_components=3)

wine_pca=pca.fit_transform(wine2_normal)

wine_pca

pca.components_

## Amount of Variance for each PCA
var=pca.explained_variance_ratio_

var

## The Cummulative variance of each pca
var1=np.cumsum(np.round(var, decimals=4)*100)

var1

pca_data=pd.DataFrame(wine_pca)

pca_data.columns=["PC1","PC2","PC3"]

pca_data

#### Here we got the 3 Principle Components now we build the clustering through these 3 points or components

plt.figure(figsize=(16,8))
sns.scatterplot(data=pca_data)
plt.show()


### After that now we explore Clustering methods #####
### First we do the Hierarchical Clustering 
### Second we do the Kmeans clustering which is non-hierarchy

### Now let us import the Libraries which are necessary for the Hierarchical CLustering 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

## after the importing the libraries now we dendrogram 

### Creating the Dendrogram for all the linkages (Single, Average,Complete)
# Dendrogram for single linkage 
dendrogram=sch.dendrogram(sch.linkage(pca_data, method="single"))
# Dendrogram for average linkage 
dendrogram=sch.dendrogram(sch.linkage(pca_data, method="average"))
# Dendrogram for complete linkage
dendrogram=sch.dendrogram(sch.linkage(pca_data, method="complete"))

# Now applying AgglomerativeClustering choosing 3 as clusters from the above dendrogram
HC = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(pca_data) 
HC.labels_

cluster_labels = pd.Series(HC.labels_)# creating a new column and assigning it to new column 

cluster_labels

wine['clusters'] = cluster_labels

wine
#Hierarchical clustering with cluster"0"
wine[wine["clusters"]==0]
#Hierarchical clustering with cluster"1"
wine[wine["clusters"]==1]
#Hierarchical clustering with cluster"2"
wine[wine["clusters"]==2]
#Hierarchical clustering with cluster"3"
wine[wine["clusters"]==3]

####2.Kmeans Clustering 
from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 
import warnings
warnings.filterwarnings('ignore')
TWSS=[]

k=list(range(1,10))

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(pca_data)
    TWSS.append(kmeans.inertia_)

TWSS

# Scree plot 
plt.plot(k, TWSS, 'ms-');plt.title("Elbow Method");plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

plt.figure(figsize=(12,8))
plt.scatter(pca_data["PC1"], pca_data["PC2"], pca_data["PC3"], cmap = plt.cm.coolwarm)
plt.show()

## Model Building to combine the pca=3 components with N_clusters=3

model = KMeans(n_clusters = 3)
model.fit(pca_data) 

clusters=model.predict(pca_data)

clusters

pca_data['cluster']=clusters

pca_data

plt.figure(figsize=(12,8))
plt.scatter(pca_data["PC1"], pca_data["PC2"], pca_data["PC3"],c=pca_data['cluster'], cmap = plt.cm.coolwarm)
plt.show()

### Kmeans Clustering with clusters "0"

pca_data[pca_data['cluster']==0]
### Kmeans Clustering with clusters "1"
pca_data[pca_data['cluster']==1]
### Kmeans Clustering with clusters "2"
pca_data[pca_data['cluster']==2]
### Kmeans Clustering with clusters "3"
pca_data[pca_data['cluster']==3]

#### Checking whether we have obtained the same number of clusters with the original data
wine_norm=scale(wine)

norm_df=pd.DataFrame(wine_norm)

norm_df

##1. Checking with original data using Hierarchical clustering

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

#dendrogram for single linkage
dendrogram = sch.dendrogram(sch.linkage(norm_df, method="single"))
#dendrogram for average linkage
dendrogram = sch.dendrogram(sch.linkage(norm_df, method="average"))
#dendrogram for complete linkage
dendrogram = sch.dendrogram(sch.linkage(norm_df, method="complete"))

# Now applying AgglomerativeClustering choosing 3 as clusters from the above dendrogram
HC = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="single").fit(norm_df)
HC

y_predict = HC.fit_predict(norm_df)

y_predict

wine["clusters_original_data"]=clusters

wine
## Hierarchical Clustering with cluster"0"
wine[wine["clusters_original_data"]==0]
## Hierarchical Clustering with cluster"1"
wine[wine["clusters_original_data"]==1]
## Hierarchical Clustering with cluster"2"
wine[wine["clusters_original_data"]==2]
## Hierarchical Clustering with cluster"3"
wine[wine["clusters_original_data"]==3]

##2. Checking with original data using KMeans Clustering
from sklearn.cluster import KMeans
# from scipy.spatial.distance import cdist 

import warnings
warnings.filterwarnings("ignore")

TWSS=[]

k=list(range(1,10))

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(pca_data)
    TWSS.append(kmeans.inertia_)

TWSS    
## Scree plot or Elbow plot 
plt.figure(figsize=(10,5));plt.plot(range(1,10), TWSS,"ms-");plt.title('Elbow Method');plt.xlabel('Optinum number of clusters');plt.show()

plt.figure(figsize=(12,8))
plt.scatter(pca_data['PC1'], pca_data['PC2'], pca_data['PC3'], cmap = plt.cm.coolwarm)
plt.show()

K_Means = KMeans( n_clusters=3, algorithm='auto', max_iter=500)
K_Means

K_Means.fit(norm_df)

clusters = K_Means.predict(norm_df)
clusters

pca_data['cluster_original_data_1']=clusters
pca_data['cluster_original_data']=clusters
pca_data
pca_data.columns

plt.figure(figsize=(12,8))
plt.scatter(pca_data['PC1'], pca_data['PC2'], pca_data['PC3'], c=pca_data['cluster_original_data_1'])
plt.show()
## Kmeans clustering with cluster "0"
pca_data[pca_data["cluster_original_data_1"]==0]
## Kmeans clustering with cluster "1"
pca_data[pca_data["cluster_original_data_1"]==1]
## Kmeans clustering with cluster "2"
pca_data[pca_data["cluster_original_data_1"]==2]
## Kmeans clustering with cluster "3"
pca_data[pca_data["cluster_original_data"]==3]

pca_data.columns

### Checking same number of clusters
### Hierarchical clustering 
wine["clusters"].value_counts()

pca_data["cluster_original_data"].value_counts()

### Kmeans clustering 
pca_data["cluster_original_data_1"].value_counts()

pca_data["cluster"].value_counts()
