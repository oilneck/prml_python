import numpy as np
import matplotlib.pyplot as plt


N = 100  #the number of noise
outlier = 20 # the number of outlier (<N)
alpha = 0.001
#sigmoid function
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

# Return the design_matrix
def make_design_matrix(phi):
    return np.vstack((np.ones(N),phi.T))

#Calculate the Parameter matrix
def make_Param_matrix(phi,T):
    PHI_T = make_design_matrix(phi)
    return np.linalg.inv(PHI_T @ PHI_T.T) @ PHI_T @ T

#Calculate new weight -> INPUT:feature-vector, target-vector, w_old  OUTPUT:w_new
def make_NEW_weight(phi,t,w_old):
    #PHI,R,H -> design matrix, weighted matrix, Hessian matrix, respectively.
    PHI_T = make_design_matrix(phi)
    y = sigmoid(w @ PHI_T)
    R = np.diag(y * (np.ones(N)-y)) + alpha * np.eye(N,N)
    H = PHI_T @ (R @ PHI_T.T)
    R_INV = np.linalg.inv(R)
    z = PHI_T.T @ w_old-R_INV @ (y-t)
    return np.linalg.inv(H) @ PHI_T @ R @ z

def decision_boundary_Least(W,x):
    W_T = W.transpose()
    return - ((W_T[0,1]-W_T[1,1]) / (W_T[0,2]-W_T[1,2])) * x - (W_T[0,0]-W_T[1,0])/(W_T[0,2]-W_T[1,2])

def decision_boundary_logistic(w,x):
    return -(w[1]/w[2])*x-(w[0]/w[2])



# Creating test data
cov = [[2.5,1], [1,0.8]]
cls1 = np.random.multivariate_normal([-2,2.5], cov, int(N/2))
cls2 = np.vstack((np.random.multivariate_normal([1,-1], cov, int(N/2)-outlier),np.random.multivariate_normal([7,-6], [[0.5,0.2],[0.2,0.5]], outlier)))
feature_phi = np.vstack((cls1,cls2))
#---------------------------------------Least Squares Method--------------------------------------------------
Teacher_T_matrix = np.vstack(([[1,0]]*cls1.shape[0],[[0,1]]*cls2.shape[0]))#1-of-K encoding
Param_matrix_W = make_Param_matrix(feature_phi,Teacher_T_matrix)

#---------------------------------------logistic regression---------------------------------------------------
target_t = np.hstack((np.ones(cls1.shape[0]),np.zeros(cls2.shape[0])))#NOTICE: target_t is not 1-of-K encoding
w = np.array([0.01,0.01,0.01])
# Updating weight vector
while True:
    w_new = make_NEW_weight(feature_phi,target_t,w)
    if np.allclose(w, w_new,rtol=0.01): break
    w = w_new


#plotting data
plt.close()
testX_cls1, testY_cls1 = np.array(cls1).T
plt.scatter(testX_cls1, testY_cls1,c='r',marker='x',label="class1",s=30,linewidth=1.5)

testX_cls2, testY_cls2 = np.array(cls2).T
plt.scatter(testX_cls2, testY_cls2, facecolor="none", edgecolor="b",label="class2",s=50,linewidth=1.5)

x = np.arange(-10,10,0.01)
y_leastsq = decision_boundary_Least(Param_matrix_W,x)
y_logistic = decision_boundary_logistic(w,x)
plt.plot(x,y_leastsq,color='purple')
plt.plot(x,y_logistic,color='lime')

plt.xlim(-6, 9)
plt.ylim(-10, 6)
plt.show()
