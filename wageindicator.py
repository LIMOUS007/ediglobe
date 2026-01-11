import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df= pd.read_csv("share-of-population-in-extreme-poverty.csv")
print(df.head())
print(df.describe())
print(df.info())
df = df[df["Share of population"].notna()]   
df = df[df["Share of population"] > 0]   
latest = df.sort_values("Share of population").drop_duplicates("Entity", keep="last") 
print(latest)
print(latest.info())
plt.figure(figsize=(14,25))
plt.barh(latest['Entity'], latest['Share of population'], color='orange')
plt.xlabel('Share of population in extreme poverty (%)')
plt.ylabel('Entity')
plt.title('Entities by Share of Population in Extreme Poverty')
plt.yticks(rotation=0)
plt.tick_params(axis='y', labelsize=5)
plt.show()
