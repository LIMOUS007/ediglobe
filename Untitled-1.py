def addition(x,y):
    print("Addition of", x, "and", y, "is:", x + y)

print(addition(5, 3))
q= lambda p, u: print("Addition of", p, "and", u, "is:", p+ u)
print(q(4, 8))
import pandas as pd
df = {
    "name": ['Aryan', 'sakshi', 'Aman', 'Rohan', 'Sahil'],
    "age": [20, 21, 19, 22, 20],
    "city": ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
}

df2 = {
    'name': ['Aryan', 'sakshi', 'Aman', 'Rohan', 'Sahil'],
    'newage': [204334, 21343, 14349, 2342, 25350],
    'newcity': ['Delasdhi', 'Mumbadasi', 'Bangaldasdore', 'Chenadsdasnai','Kolkadasdasta']
}
df = pd.DataFrame(df)
df2 = pd.DataFrame(df2)
print(df)
print(df2)
df3 = pd.merge(df, df2, on='name')
print(df3)