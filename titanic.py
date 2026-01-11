import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
titanic = sns.load_dataset('titanic')
print(titanic)
print(titanic.info())
corMat = titanic.corr(numeric_only=True)
print(corMat)
sns.heatmap(corMat, annot=True, cmap='coolwarm')
plt.show()
from scipy.stats import kurtosis
from scipy.stats import skew
sns.kdeplot(titanic['age'].dropna(), fill=True)
plt.title('KDE Plot of Age')
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()
print("Kurtosis of Age:", kurtosis(titanic['age'].dropna()))
print("Skewness of Age:", skew(titanic['age'].dropna()))
