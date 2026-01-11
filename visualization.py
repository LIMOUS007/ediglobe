import re
import matplotlib.pyplot as plt 

sentence = "My Contact number is 12345-67890 and my email is xyz@gmail.com"
numberPattern = r"\d{5}-\d{5}"
numMatch = re.search(numberPattern, sentence)
print("Contact Number:", numMatch.group(0))
if numMatch:
    print('Contact number found:', numMatch.group())

month = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
savings = [1000, 1500, 2000, 2500, 3000]
savings2 = [1200, 1600, 2100, 2400, 2900]

plt.plot(month, savings2, marker="o", color='red', label='Savings 2')
plt.plot(month, savings, marker="o", color="blue", label='Savings 1')
plt.xlabel('Months')
plt.ylabel('Savings')   
plt.title('Monthly Savings Comparison')
plt.grid()
plt.show()

products = ['Product A', 'Product B', 'Product C']
sales = [150, 200, 300]
plt.bar(products, sales, color='green')
plt.show()

marks = [85, 90, 78, 92, 88, 2 ,72, 24,24,2 ,42,4,24,2,42,4,24,2,42,4,2,42,4,2,42,4,2,42,4,2,42,4]
plt.hist(marks, bins=10, color='purple', edgecolor='black', rwidth=0.7)
plt.xlabel('Marks')
plt.ylabel('Frequency')
plt.title('Marks Distribution')
plt.show()