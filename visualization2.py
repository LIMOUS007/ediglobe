import re
import matplotlib.pyplot as plt 
h = [150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200]
w = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
plt.scatter(h, w)
plt.show()
products = ['A','B','C','D'] # Categorical Data
sales = [120,340,210,180]
plt.pie(sales, labels= products, autopct='%0.1f%%')
plt.show() 
plt.barh(products, sales, color='orange')
plt.show()