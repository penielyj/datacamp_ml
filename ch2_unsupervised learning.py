#!/usr/bin/env python
# coding: utf-8

# ## Unsupervised Learning

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

import warnings
warnings.filterwarnings(action='ignore')


# ### 1) clustering for dataset exploration

# In[ ]:


from sklearn.cluster import KMeans
model = KMeans(n_clusters = 3)
model.fit(samples)

labels = model.predict(samples)
print(label)
print(model.inertia_) #관성을 통한 평가방법


# #### transforming features for better clusterings
# 
# - 각 변수의 분산을 최소화 하기 위해, 클러스터링 전에 표준화 진행
# - 표준화: StandardScaler

# In[3]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(samples)
StandardScaler(copy=True, with_mean=True, with_std=True)
samples_scaled = scaler.transform(samples)


# In[ ]:


#파이프라인
scaler = StandardScaler()
model = KMeans(n_clusters = 3)
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(samples)
labels = pipeline.predict(samples)


# ### 2) Visualization with hierarchical clustering and t_SNE

# In[ ]:


#Hierarchical clustering 
from scipy.cluster.hierarchy import linkage, dendrogram
mergings = linkage(samples, method = 'complete')
dendrogram(mergings, labels = country_names,
          leaf_rataion=90,
          leaf_font_size=6)
plt.show()


# #### t_SNE
# - 고차원 공간의 샘플을 2차원 or 3차원 공간으로 매핑하여 시각화
# - has a fit_transform() method
# - can't extend the map to include new data samples -> must start over each time!
# - t-sne features are different every time

# In[ ]:


from sklearn.manifold import TSNE
model = TSEN(learning_rate=100)

transformed = model.fit_transform(samples)

xs = transformed[:,0]
ys = transformed[:,1]

plt.scatter(xs, ys, c=sepcies)

for x,y,sample in zip(xs, ys, sample):
    plt.annotate(sample, (x,y), fontsize=5, alpha=0.75)

plt.show()


# In[ ]:


from scipy.cluster.hierarchy import linkage, dendrogram


# In[ ]:


#스케일링
#from sklearn.preprocessing import normalize
#normalized_sample = normalize(sample)
mergings = linkage(samples, method='complete')
dendrogram(mergings,
           labels = country_names,
           leaf_rotation = 90,
           left_font_size=6)
plt.show()


# ### cluster labels in hierarchical clustering

# In[ ]:


#덴드로그램의 y축은 병합 클러스터 사이의 거리를 인코딩한다.
#linkage
## complete: distance between clusters in max.

from scipy.cluster.hierarchy import linkage
mergings = linkage(samples, method = 'complete')
from scipy.cluster.hierarchy import fcluster
labels = fcluster(mergings, 15, criterion='distance')
print(labels)


# In[ ]:


pairs = pd.DataFrame({'label':labels, 'countries':country_names})
print(pairs.sort_values('labels'))


# ## 3) PCA(=Principal Component Analysis)

# ### Dimension reduction

# In[ ]:


from sklearn.decomposition import PCA
model = PCA()
model.fit(samples)
transformed = model.transform(samples)


# #### Intrinsic dimension 

# In[ ]:


#plotting the variances of PCA features

pca = PCA()
pca.fit(samples)
feature = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()


# In[ ]:


#dimension reduction with PCA
pca = PCA(n_components=2)
pca.fit(samples)

transformed = pca.transform(samples)
print(transformed.shape)


# #### truncatedSVD and csr_matrix

# In[ ]:


from sklearn,decomposition import TruncatedSVD
model = TruncatedSVD(n_components=3)
model.fit(documents)
transformed = model.transform(documents)


# ## 4) Non-negative matrix factorization(NMF)

# In[ ]:


from sklearn.decomposition import NMF
model = NMF(n_components=2)
model.fit(samples)

nmf_features = model.transform(samples)

print(model.components_)


# #### applying NMF to the articles

# In[ ]:


bitmap = sample.reshape((2,3))

plt.imshow(bitmap, camp='gray', interpolation='nearset')
plt.show()

