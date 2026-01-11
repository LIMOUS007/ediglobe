import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
nig = pd.read_csv("book2.csv")
print(nig.head())
print(nig.info())
print(nig.describe())
nig.dropna(inplace=True)
y = nig['Air transport, passengers carried']
x = nig.drop('Air transport, passengers carried', axis=1)