import pandas as pd
data = {
    "name" : ['Aryan', 'sakshi', 'Aman', 'Rohan', 'Sahil'],
    "age" : [20, 21, 19, 22, 20],
    "city" : ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
}
print (data)
df = pd.DataFrame(data)
print(df)
df1 = pd.read_csv("cars.csv")
print(df1)
df2 = pd.read_excel("sample2.xlsx")
print(df2)
print(df2.head(10))
print(df2.tail())   
print(df2.describe())
print(df2.info())
print(df2.shape)
