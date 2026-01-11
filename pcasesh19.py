from sklearn import datasets
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
dataset = load_digits()
print (dataset)
print (dataset.keys())
df = pd.DataFrame(dataset.data, columns = dataset.feature_names)
print (df)
df['Target'] = dataset.target
print (df)
print("target is: ", dataset.target[0])
print (dataset.data[0].reshape(8,8))
#plt.gray()
#plt.matshow(dataset.data[0].reshape(8,8))
#plt.show()
num_image = 10
plt.figure(figsize= (7,2))
for i in range (num_image):
    plt.subplot(2,10,i+1)
    plt.imshow(dataset.images[i], cmap = "gray")
    plt.title(f"lable: {dataset.target[i]}")
    plt.axis('off')
#plt.tight_layout()
#plt.show()
x = df.iloc[:, :-1]
print(x)
y = dataset.target
scalar = StandardScaler()
x_scaled = scalar.fit_transform(x)
print (x_scaled)
print(pd.DataFrame(x_scaled))
x1 = x_scaled.T
print (pd.DataFrame(x1))
cov_mat = np.cov(x_scaled.T)
print (pd.DataFrame(cov_mat))
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print(pd.DataFrame(eig_vecs))
total = sum(eig_vals)
var_exp = [(i/total)*100 for i in sorted(eig_vals,reverse=True)]
print(var_exp)
cum_var_exp = np.cumsum(var_exp)
print(pd.DataFrame(cum_var_exp))
plt.show()
plt.bar(range(len(var_exp)), var_exp, label='individual explained variance',color='g')
plt.step(range(len(cum_var_exp)),cum_var_exp, label='Cumulative explained variance')
plt.legend()
plt.show()

