import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = sns.load_dataset('mpg')

print("First 5 rows of the dataset:")
print(df.head())

print("\nMean of the 'weight' column:")
print(df["weight"].mean())

print("\nMedian of the 'weight' column:")
print(df["weight"].median())

print("\nMode of the 'weight' column:")
print(df["weight"].mode()[0])

print("\nStandard deviation of the 'weight' column:")
print(df["weight"].std())

print("\nVariance of the 'weight' column:")
print(df["weight"].var())

print("\nMinimum value in the 'weight' column:")
print(df["weight"].min())

print("\nNumber of unique values in the 'weight' column:")
print(df["weight"].nunique())

print("\nRange (max - min) of the 'weight' column:")
print(np.ptp(df["weight"]))

age = [23, 45, 56, 78, 89, 90, 34, 23, 45, 67, 89, 90, 34, 23, 45, 67]
Q1 = np.percentile(df["weight"], 25)
Q3 = np.percentile(df["weight"], 75)
IQR = Q3 - Q1
print("\nInterquartile Range (IQR) of the 'weight' list:")
print(IQR)
Upper_fence = Q3 + 1.5 * IQR
Lower_fence = Q1 - 1.5 * IQR
print("Upper fence:", Upper_fence)
print("Lower fence:", Lower_fence)
x = df["weight"].copy()
print("List before removing outliers:")
print(df["weight"])
for i in x:
    if i > Upper_fence or i < Lower_fence:
        print("Outlier detected:", i)
        df["weight"].remove(i)
print("List after removing outliers:")
print(df["weight"]) 
plt.boxplot(df["weight"])
plt.title("Box plot of weight")
plt.show()

