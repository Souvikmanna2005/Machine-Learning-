# LinearRegression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#1.Prepare data(Example:Hours studied vs. Test Score)
X=np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
Y=np.array([2,4,5,4,5,7,8,9,10,12])

# print(X)
# print(Y)
#2.Create and train the model
model=LinearRegression()
model.fit(X,Y)

#3.Make predictions
y_pred=model.predict(X)

#4.Visualize the result
plt.scatter(X,Y,color='blue',label='Actual Data') #scatter plot of original points
plt.plot(X,y_pred,color='red',linewidth=2,label='Regression Line') #Line of best fit
plt.title('Simple Linear Regression Example')
plt.xlabel('Independent Variable(X)')
plt.ylabel('Dependent Variable(Y)')
plt.legend()         
plt.show()

# Print the model parameters
print(f"Slope (Coefficient): {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

#OUTPUT
# Slope (Coefficient): 1.0060606060606059
# Intercept: 1.0666666666666673