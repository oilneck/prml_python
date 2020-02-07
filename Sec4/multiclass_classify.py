import numpy as np
import matplotlib.pyplot as plt
from classifier.softmax_regression import Softmax_Regression
from base_module.poly_feature import Poly_Feature
N = 150  # The number of test data
K = 3    # The number of class



'''-------Training Data------'''
cov = [[1.0,0.8], [0.8,1.0]]
cls1 = np.random.multivariate_normal([-2,2], cov, int(N/K))
cls2 = np.random.multivariate_normal([0,0], cov, int(N/K))
cls3 = np.random.multivariate_normal([2,-2], cov, int(N/K))
feature = Poly_Feature(1)
PHI_train = feature.transform(np.vstack((cls1,cls2,cls3)))
#Creating Label Matrix T
T_train = np.vstack(([[1,0,0]]*cls1.shape[0],[[0,1,0]]*cls2.shape[0],[[0,0,1]]*cls3.shape[0]))

'''----------Test Data------------'''
x = np.arange(-10,10,0.01)
y = np.arange(-10,10,0.01)
X,Y = np.meshgrid(x,y)
test_x = np.array([X.ravel(), Y.ravel()]).reshape(2,-1).T
X_test = feature.transform(test_x)

'''-------logistic regression-------'''
model = Softmax_Regression()
model.fit(PHI_train,T_train)
Z = model.predict(X_test)


#Plotting data
plt.scatter(cls1.T[0],cls1.T[1],c='r',marker='x',label="class1",s=40,linewidth=1.5)
plt.scatter(cls2.T[0], cls2.T[1] , color='limegreen',marker='+',linewidth=1.5,s=60)
plt.scatter(cls3.T[0], cls3.T[1] ,facecolor="none", edgecolor="b",label="class2",s=50,linewidth=1.5)
#Drawing area
plt.contourf(X,Y, Z.reshape(X.shape), alpha=0.2, levels=np.array([0., 0.5, 1.5, 2.]),colors=['lightcoral','lime','cornflowerblue'])

plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.show()
