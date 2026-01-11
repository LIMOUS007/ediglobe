import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
x, y= make_blobs(n_samples=50, centers=2, n_features=2, random_state=3)
print (x)
print (y)
df = pd.DataFrame(x, y)
print(df.head)
df = df.rename(columns= {0: 'Feature 1', 1 : 'Feature 2'})
print(df.head())
plt.scatter(x[:,0], x[:,1], label = y)
plt.xlabel('X: Feature 1')
plt.ylabel('Y: Feature 2')
plt.show()
link = linkage(df, method='ward', metric='euclidean')
plt.figure(figsize= (10,7))
plt.title('Dendogram')
den = dendrogram(link)
plt.show()
model = AgglomerativeClustering(linkage='ward', metric = 'euclidean')
y_pred = model.fit_predict(df)
plt.scatter(df['Feature 1'], df['Feature 2'], c = y_pred, cmap = "plasma")
plt.show() 