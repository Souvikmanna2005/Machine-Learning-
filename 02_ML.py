"""
Write a python code compute Pearson Correlation between two specific 
columns. Graphically plot the heatmap.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
'Math':[78,85,96,80,86],
'Science':[88,90,94,82,89],
'English':[72,75,78,70,74]
}
df=pd.DataFrame(data)

#Compute Pearson Correlation between two specific columns
#The default method for .corr() is 'pearson
correlation=df['Math'].corr(df['Science'])
print(f"Pearson Correlation between Math and Science:{correlation:.4f}")

plt.figure(figsize=(8,6))
corr_matrix=df.corr()
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt=".2f")

plt.title('Pearson Correlation Heatmap')
plt.show()   

#output : Pearson Correlation between Math and Science:0.7990