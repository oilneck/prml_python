import numpy as np
import matplotlib.pyplot as plt
from classifier.logistic_regression import Logistic_Regression

N = 100  #the number of noise
outlier = 20 # the number of outlier (<N)

# Return the design_matrix
def make_design_matrix(phi):
    return np.vstack((np.ones(len(phi)),phi.T)).T

#Calculate the Parameter matrix
def make_Param_matrix(PHI,T):
    PHI_T = PHI.T
    return np.linalg.inv(PHI_T @ PHI) @ PHI_T @ T

def decision_boundary_Least(W,x):
    W_T = W.transpose()
    return - ((W_T[0,1]-W_T[1,1]) / (W_T[0,2]-W_T[1,2])) * x - (W_T[0,0]-W_T[1,0])/(W_T[0,2]-W_T[1,2])





# Creating test data
cov = [[2.5,1], [1,0.8]]
cls1 = np.random.multivariate_normal([-2,2.5], cov, int(N/2))
cls2 = np.vstack((np.random.multivariate_normal([1,-1], cov, int(N/2)-outlier),np.random.multivariate_normal([7,-6], [[0.5,0.2],[0.2,0.5]], outlier)))
train_X = make_design_matrix(np.vstack((cls1,cls2)))
#---------------------------------------Least Squares Method--------------------------------------------------
Teacher_T_matrix = np.vstack(([[1,0]]*cls1.shape[0],[[0,1]]*cls2.shape[0]))#1-of-K encoding
Param_matrix_W = make_Param_matrix(train_X,Teacher_T_matrix)

#---------------------------------------logistic regression---------------------------------------------------
train_t = np.hstack((np.ones(cls1.shape[0]),np.zeros(cls2.shape[0])))#NOTICE: target_t is not 1-of-K encoding
logistic_regression = Logistic_Regression()
logistic_regression.fit(train_X,train_t)


#plotting data
plt.close()
testX_cls1, testY_cls1 = np.array(cls1).T
plt.scatter(testX_cls1, testY_cls1,c='r',marker='x',label="class1",s=30,linewidth=1.5)

testX_cls2, testY_cls2 = np.array(cls2).T
plt.scatter(testX_cls2, testY_cls2, facecolor="none", edgecolor="b",label="class2",s=50,linewidth=1.5)

x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)
X,Y = np.meshgrid(x,y)


y_leastsq = decision_boundary_Least(Param_matrix_W,x)
plt.plot(x,y_leastsq,color='purple')

test_x = np.array([X.ravel(), Y.ravel()]).reshape(2,-1).T
z_lr = logistic_regression.predict(make_design_matrix(test_x))
plt.contour(X,Y,z_lr.reshape(X.shape),[0],colors='lime')

plt.xlim(-6, 9)
plt.ylim(-10, 6)
plt.show()
