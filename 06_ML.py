#LOGISTIC REGRESSION

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

#1.Dataset:Hours Studied(X) vs pass/fail (Y)
X=np.array([0.5,0.75,1.0,1.25,1.5,1.75,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,4.0,4.25,4.5,
            4.75,5.0,5.5]).reshape(-1,1)
Y=np.array([0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1])

# print(X)
# print(Y)

#.Train Model
model=LogisticRegression()
model.fit(X,Y)
y_pred=model.predict(X)

#3.Visualization
plt.figure(figsize=(12,5))

#plot 1:Logistic Regression Sigmoid Curve
plt.subplot(1,2,1)
X_range=np.linspace(0,6,100).reshape(-1,1)
#print(X_range)
Y_prob=model.predict_proba(X_range)[:,1]
#print(Y_prob)
plt.scatter(X,Y,color='red',label='Actual Data')
plt.plot(X_range,Y_prob,color='blue',label='Probability Curve')
plt.axhline(0.5,color='black',linestyle='--') #Decision Threshold
plt.title('Study Hours Vs Pass Probability')
plt.xlabel('Hours Studied')
plt.ylabel('Pass Probability')
plt.legend()

#plot 2:Confusion Matrix Heatmap
plt.subplot(1,2,2)
cm=confusion_matrix(Y,y_pred)
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['Fail','Pass'],
            yticklabels=['Fail','Pass'])

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()

#4.print Detailed Metrix
print(classification_report(Y,y_pred))

accuracy=accuracy_score(Y,y_pred)
precision=precision_score(Y,y_pred)
recall=recall_score(Y,y_pred)
f1=f1_score(Y,y_pred)


print(f"Accuracy :{accuracy:.2f}")
print(f"Precision :{precision:.2f}")
print(f"Recall :{recall:.2f}")
print(f"F1 score :{f1:.2f}")


#OUTPUT
#               precision    recall  f1-score   support

#            0       0.80      0.80      0.80        10
#            1       0.80      0.80      0.80        10

#     accuracy                           0.80        20
#    macro avg       0.80      0.80      0.80        20
# weighted avg       0.80      0.80      0.80        20

# Accuracy :0.80
# Precision :0.80
# Recall :0.80
# F1 score :0.80