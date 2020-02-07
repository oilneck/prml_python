import numpy as np
import matplotlib.pyplot as plt
from classifier.least_squares_classifier import Least_Squares_Classifier
from classifier.logistic_regression import Logistic_Regression
from base_module.poly_feature import Poly_Feature

N = 100  #the number of noise
outlier = 20 # the number of outlier (<N)



# Creating training data
cov = [[2.5,1], [1,0.8]]
cls1 = np.random.multivariate_normal([-2,2.5], cov, int(N/2))
cls2 = np.vstack((np.random.multivariate_normal([1,-1], cov, int(N/2)-outlier),np.random.multivariate_normal([7,-6], [[0.5,0.2],[0.2,0.5]], outlier)))
# Create the design_matrix
feature = Poly_Feature(1)
X_train = feature.transform(np.vstack((cls1,cls2)))


# Creating test data
X,Y = np.meshgrid(np.linspace(-10,10,100),np.linspace(-10,10,100))
test_x = np.array([X.ravel(), Y.ravel()]).reshape(2,-1).T
X_test = feature.transform(test_x)


'''----------Least Squares Method----------'''
#1-of-K encoding
Teacher_T_matrix = np.vstack(([[1,0]]*cls1.shape[0],[[0,1]]*cls2.shape[0]))
least_squares = Least_Squares_Classifier()
least_squares.fit(X_train,Teacher_T_matrix)
z_ls = least_squares.predict(X_test)


'''------------logistic regression-------------'''
#NOTICE: train_t is not 1-of-K encoding
train_t = np.hstack((np.ones(cls1.shape[0]),np.zeros(cls2.shape[0])))
logistic_regression = Logistic_Regression()
logistic_regression.fit(X_train,train_t)
z_lr = logistic_regression.predict(X_test)


#plotting training data
plt.scatter(cls1.T[0],cls1.T[1],c='r',marker='x',label="class1",s=30,linewidth=1.5)
plt.scatter(cls2.T[0],cls2.T[1], facecolor="none", edgecolor="b",label="class2",s=50,linewidth=1.5)

#plotting test data
plt.contour(X,Y,z_ls.reshape(X.shape),[0],colors='purple')
plt.contour(X,Y,z_lr.reshape(X.shape),[0],colors='lime')

plt.xlim(-6, 9)
plt.ylim(-10, 6)
plt.show()
