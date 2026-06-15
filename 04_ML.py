"""
Write a python code to develop confusion matrix, Find the confusion matrix
based metrics. 
"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score

y_true=[0,1,0,1,0,1,1,0,1,0] #ground truth
y_pred=[0,1,0,0,0,1,1,1,1,0] #models output

#2.compute confusion matrix
cm=confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n",cm)

#3.individual Metrics calculation
print(f"Accuracy:{accuracy_score(y_true,y_pred):.2f}")
print(f"Precision:{precision_score(y_true, y_pred):.2f}")
print(f"Recall:{recall_score(y_true, y_pred):.2f}")
print(f"F1_score:{f1_score(y_true, y_pred):.2f}")

#4.comprehensive classification report
#This prints precision,recall,and f1_score for every class at once
print("\nClassification Report:\n",classification_report(y_true,y_pred))

#5.Visualizing the confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Negative','Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix Visualization")
plt.show()

#output

# Confusion Matrix:
#  [[4 1]
#  [1 4]]
# Accuracy:0.80
# Precision:0.80
# Recall:0.80
# F1_score:0.80

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.80      0.80      0.80         5
#            1       0.80      0.80      0.80         5

#     accuracy                           0.80        10
#    macro avg       0.80      0.80      0.80        10
# weighted avg       0.80      0.80      0.80        10