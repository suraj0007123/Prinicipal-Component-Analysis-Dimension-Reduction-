## Importing All Necessary Libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from scipy.spatial.distance import cdist 
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
import warnings 
warnings.filterwarnings("ignore")


### Importing data for PCA and hierarchical clustering and Kmeans analysis
heartdata = pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\dimension reducation\Datasets_PCA\heart disease.csv")

heartdata.describe() ## To know stats about the data

heartdata.info() ## To know information about the data

###### Dropping uneccessary column #####
heartdata1 = heartdata.drop(["target"], axis = 1)

###### Identifying duplicates #########
duplicate=heartdata1.duplicated()
duplicate
sum(duplicate)
data1=heartdata1.drop_duplicates()

##### The Outliers Treatment ###########
data1.columns
sns.boxplot(data1.age)

data1.columns
sns.boxplot(data1.sex)

sns.boxplot(data1.cp)

sns.boxplot(data1.trestbps)
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['trestbps'])
df_t=winsor.fit_transform(data1[['trestbps']])
sns.boxplot(df_t.trestbps)

data1.columns
sns.boxplot(data1.chol)
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['chol'])
df_t=winsor.fit_transform(data1[['chol']])
sns.boxplot(df_t.chol)

sns.boxplot(data1.fbs)
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['fbs'])
df_t=winsor.fit_transform(data1[['fbs']])
sns.boxplot(df_t.fbs)

sns.boxplot(data1.restecg)

sns.boxplot(data1.thalach)
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['thalach'])
df_t=winsor.fit_transform(data1[['thalach']])
sns.boxplot(df_t.thalach)

sns.boxplot(data1.exang)

sns.boxplot(data1.oldpeak)
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['oldpeak'])
df_t=winsor.fit_transform(data1[['oldpeak']])
sns.boxplot(df_t.oldpeak)

sns.boxplot(data1.slope)

sns.boxplot(data1.ca)
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['ca'])
df_t=winsor.fit_transform(data1[['ca']])
sns.boxplot(df_t.ca)

sns.boxplot(data1.thal)
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['thal'])
df_t=winsor.fit_transform(data1[['thal']])
sns.boxplot(df_t.thal)

data1.isna().sum()

data1.dtypes

data1.shape

# Considering only numerical data 
data1.data = data1.iloc[:, 0:]

# Normalizing the numerical data 
data1_normal = scale(data1.data)
data1_normal
pca = PCA(n_components = 10)
pca_values = pca.fit_transform(data1_normal)

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
pca_data.columns = "Z1", "Z2", "Z3","Z4","Z5","Z6","Z7","Z8","Z9","Z10"
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
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram on HeartDisease Data');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()

z = linkage(pca_data, method = "average", metric = "euclidean")
# Dendrogram on average linkage method
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram on HeartDisease Data');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()

z = linkage(pca_data, method = "complete", metric = "euclidean")
# Dendrogram on complete linkage method
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram on HeartDisease Data');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()

h_complete = AgglomerativeClustering(n_clusters = 10, linkage = 'complete', affinity = "euclidean").fit(pca_data) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

data1['clusters'] = cluster_labels # creating a new column and assigning it to new column 

data1
## Hierarchical Clustering on cluster "0"
data1[data1['clusters']==0]
## Hierarchical Clustering on cluster "1"
data1[data1['clusters']==1]
## Hierarchical Clustering on cluster "2"
data1[data1['clusters']==2]
## Hierarchical Clustering on cluster "3"
data1[data1['clusters']==3]
## Hierarchical Clustering on cluster "4"
data1[data1['clusters']==4]
## Hierarchical Clustering on cluster "5"
data1[data1['clusters']==5]
## Hierarchical Clustering on cluster "6"
data1[data1['clusters']==6]
## Hierarchical Clustering on cluster "7"
data1[data1['clusters']==7]
## Hierarchical Clustering on cluster "8"
data1[data1['clusters']==8]
## Hierarchical Clustering on cluster "9"
data1[data1['clusters']==9]
## Hierarchical Clustering on cluster "10"
data1[data1['clusters']==10]

#### KMeans Clustering ######
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 
import warnings
warnings.filterwarnings("ignore")

###### scree plot or elbow curve ############
TWSS=[]

k=list(range(2,12))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(pca_data)
    TWSS.append(kmeans.inertia_)
TWSS

# Scree plot 
plt.figure(figsize=(10,5));plt.title('Elbow Method on HeartDisease Data');plt.plot(k, TWSS, 'ms-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")


# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 10)
model.fit(pca_data)

model.labels_ # getting the labels of clusters assigned to each row 
clusters = pd.Series(model.labels_)  # converting numpy array into pandas series object 
pca_data['cluster'] = clusters # creating a  new column and assigning it to new column 
pca_data

pca_data.columns # To know columns names after assigning the cluster column


## Kmeans clustering on clusters "0"
pca_data[pca_data['cluster']==0]

## Kmeans clustering on clusters "1"
pca_data[pca_data['cluster']==1]

## Kmeans clustering on clusters "2"
pca_data[pca_data['cluster']==2]

## Kmeans clustering on clusters "3"
pca_data[pca_data['cluster']==3]

## Kmeans clustering on clusters "4"
pca_data[pca_data['cluster']==4]

## Kmeans clustering on clusters "5"
pca_data[pca_data['cluster']==5]

## Kmeans clustering on clusters "6"
pca_data[pca_data['cluster']==6]

## Kmeans clustering on clusters "7"
pca_data[pca_data['cluster']==7]

## Kmeans clustering on clusters "8"
pca_data[pca_data['cluster']==8]

## Kmeans clustering on clusters "9"
pca_data[pca_data['cluster']==9]

## Kmeans clustering on clusters "10"
pca_data[pca_data['cluster']==10]




#### Checking whether we have obtained same number of clusters with the original data.
data1_normal

norm_df=pd.DataFrame(data1_normal)

norm_df

### Checking with original data using Hierarchical clustering 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

z = linkage(norm_df, method = "single", metric = "euclidean")
# Dendrogram on single linkage method
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram on HeartDisease Data');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()

z = linkage(norm_df, method = "average", metric = "euclidean")
# Dendrogram on average linkage method 
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram on HeartDisease Data');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()

z = linkage(norm_df, method = "complete", metric = "euclidean")
# Dendrogram on complete linkage method
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram on HeartDisease Data');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()

plt.figure(figsize=(12,8))
sns.scatterplot(data=norm_df)
plt.show()

h_complete = AgglomerativeClustering(n_clusters = 10, linkage = 'complete', affinity = "euclidean").fit(norm_df) 
h_complete.labels_

clusters = pd.Series(h_complete.labels_)

data1['cluster_original_data'] = clusters

data1

## Hierarchical clustering with cluster "0"
data1[data1["cluster_original_data"]==0]

## Hierarchical clustering with cluster "1"
data1[data1["cluster_original_data"]==1]

## Hierarchical clustering with cluster "2"
data1[data1["cluster_original_data"]==2]

## Hierarchical clustering with cluster "3"
data1[data1["cluster_original_data"]==3]

## Hierarchical clustering with cluster "3"
data1[data1["cluster_original_data"]==4]

## Hierarchical clustering with cluster "3"
data1[data1["cluster_original_data"]==5]

## Hierarchical clustering with cluster "3"
data1[data1["cluster_original_data"]==6]

## Hierarchical clustering with cluster "3"
data1[data1["cluster_original_data"]==7]

## Hierarchical clustering with cluster "3"
data1[data1["cluster_original_data"]==8]

## Hierarchical clustering with cluster "3"
data1[data1["cluster_original_data"]==9]

## Hierarchical clustering with cluster "3"
data1[data1["cluster_original_data"]==10]


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

k=list(range(2,12))

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(pca_data)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot
plt.figure(figsize=(10,5));plt.plot(k, TWSS, 'cs-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 10)
model.fit(norm_df)

model.labels_ # getting the labels of clusters assigned to each row 
clusters = pd.Series(model.labels_)  # converting numpy array into pandas series object 
pca_data['cluster_original_data'] = clusters
pca_data['cluster_original_data_1'] = clusters # creating a  new column and assigning it to new column 

pca_data

## kmeans clustering with clusters "0"
pca_data[pca_data['cluster_original_data_1']==0]

## kmeans clustering with clusters "1"
pca_data[pca_data['cluster_original_data_1']==1]

## kmeans clustering with clusters "2"
pca_data[pca_data['cluster_original_data_1']==2]

## kmeans clustering with clusters "3"
pca_data[pca_data['cluster_original_data_1']==3]

## kmeans clustering with clusters "4"
pca_data[pca_data['cluster_original_data_1']==4]

## kmeans clustering with clusters "5"
pca_data[pca_data['cluster_original_data_1']==5]


## kmeans clustering with clusters "6"
pca_data[pca_data['cluster_original_data_1']==6]


## kmeans clustering with clusters "7"
pca_data[pca_data['cluster_original_data_1']==7]

## kmeans clustering with clusters "8"
pca_data[pca_data['cluster_original_data_1']==8]


## kmeans clustering with clusters "8"
pca_data[pca_data['cluster_original_data_1']==9]


## kmeans clustering with clusters "8"
pca_data[pca_data['cluster_original_data_1']==10]

### Checking same number of clusters 
## Hierarchical clustering 
data1['clusters'].value_counts()

pca_data['cluster_original_data'].value_counts()


#### Kmeans clustering 
pca_data['cluster_original_data_1'].value_counts()

pca_data['cluster'].value_counts()





