"""
Write a python code to partition the dataset into Training , Validation, and 
Testing. 
"""

from sklearn.model_selection import train_test_split

X=list(range(100))
Y=list(range(100))
#step 1: split into train (70%) and temp (30%)
X_train,X_temp,Y_train,Y_temp=train_test_split(X,Y,test_size=0.3,random_state=42)

#step 2: split temp into validation (15%) and test (15%)
X_val,X_test,Y_val,Y_test=train_test_split(X_temp,Y_temp,test_size=0.5,random_state=42)

print(f"Train size:{len(X_train)}")
print(f"Validation size:{len(X_val)}")
print(f"Test size:{len(X_test)}")    


#Another way to partition
#1.split into train(80%) and Test(20%)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

#2.split train into train(75% of 80% =60%)and validation(25% of 80%=20%)
X_train,X_val,Y_train,Y_val=train_test_split(X_train,Y_train,test_size=0.25,random_state=42)

print(f"Train size:{len(X_train)}")
print(f"Validation size:{len(X_val)}")
print(f"Test size:{len(X_test)}")  

#output : 

# Train size:70
# Validation size:15
# Test size:15
# Train size:60
# Validation size:20
# Test size:20