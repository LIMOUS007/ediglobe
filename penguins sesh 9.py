import seaborn as sns   
from scipy import stats 
import numpy as np
from scipy.stats import f
penguins = sns.load_dataset("penguins").dropna()
print(penguins)     
print(stats.ttest_1samp(penguins['flipper_length_mm'], popmean = 200)) 
male = penguins[penguins['sex'] == 'Male']['flipper_length_mm']
female = penguins[penguins['sex'] == 'Female']["flipper_length_mm"]
print(stats.ttest_ind(male, female, equal_var = False))
male = penguins[penguins['sex'] == 'Male']['body_mass_g']
female = penguins[penguins['sex'] == 'Female']["body_mass_g"]
print(stats.ttest_ind(male, female, equal_var = False))
#seash 9 over
A = [25,30,28,35,40]
B = [20,22,23,24,25]
varA, varB = np.var(A, ddof = 1), np.var(B, ddof = 1)
f_stat = varA/varB
df1, df2 = len(A)-1, len(B)-1
print (f_stat)
p_value = 1-f.cdf(f_stat, df1, df2)
print (p_value)
data = np.array([[30,10],[25,15]])
from scipy.stats import chi2_contingency
print (chi2_contingency(data))
x = [85,90,88,75,95]
y = [70,65,80,72,68]
z = [88,85,90,92,87]
from scipy.stats import f_oneway
print (f_oneway(x,y,z))
from scipy.stats import norm
sample = [72, 70, 68, 65, 74, 69]
mean_samp = np.mean(sample)
pop_mean = 70
pop_std = 3
n = len(sample)
z = (mean_samp - pop_mean) / (pop_std/np.sqrt(n))
print (z)
pVal = 2 * (1-norm.cdf(abs(z)))
print (z)
print (pVal)
 