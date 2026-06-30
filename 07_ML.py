#KNN

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#1.Dataset: Features(x,y) and classes (0 or 1)
data=list(zip([4,5,10,4,3,11,14,8,10],
              [21,19,24,17,16,25,24,22,21]))
classes=[0,0,1,0,0,1,1,0,1]

#2.Instantiate and fit the KNN mode(k=3)
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(data,classes)

#3.Predict new point
new_point=[[8,21]]
prediction=knn.predict(new_point)
print(f"Prediction:{prediction[0]}")

#4.Visualize
plt.scatter([p[0] for p in data],[p[1] for p in data],c=classes)
plt.scatter(new_point[0][0],new_point[0][1],c='red',marker='+')
plt.show()