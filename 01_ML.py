"""
Write a python code to compute the pairwise pearson correlation between 
columns. Graphically plot the heatmap.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
        'Hours_Studied':[10,8,15,18,5,20,12,14,7,11],
        'Test_Scores': [85,78,92,95,60,98,88,90,72,82],
        'Caffeine_intake':[2,1,3,4,1,5,2,3,1,2]
}
df=pd.DataFrame(data)

#Compute Pearson Correlation between two specific columns
#The default method for .corr() is 'pearson
correlation=df['Hours_Studied'].corr(df['Test_Scores'])
print(f"Pearson Correlation between Hours Studied and Test Scores:{correlation:.4f}")

plt.figure(figsize=(8,6))
corr_matrix=df.corr()
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt=".2f")

plt.title('Pearson Correlation Heatmap')
plt.show()

#output : Pearson Correlation between Hours Studied and Test Scores:0.9351