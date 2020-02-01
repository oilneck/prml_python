import numpy as np
import matplotlib.pyplot as plt

N = 150  # The number of test data
D = 2   # Dimension
K = 3    # The number of class


def sigmoid(a):
    return 1 / (1 + np.exp(-a))

# Return the design_matrix -> OUTPUT:PHI_T
def make_design_matrix_trans(phi):
    return np.vstack((np.ones(N),phi.T))

# Soft-max function
def soft_max(X):
    MAX_val = np.max(X, axis=-1, keepdims=True)
    return np.exp(X-MAX_val)/np.sum(np.exp(X-MAX_val), axis = 1, keepdims = True)

def diag_Hesse_matrix(PHI,Y,cls_k):
    R = np.diag((Y[:,cls_k] *(1-Y[:,cls_k])))
    return PHI.T @ R @ PHI

def decision_boundary(x1, W_t, c1, c2):
    return (- ((W_t[c1,1]-W_t[c2,1]) / (W_t[c1,2]-W_t[c2,2]))) * x1 - ((W_t[c1,0]-W_t[c2,0]) / (W_t[c1,2]-W_t[c2,2]))


# Creating test data
cov = [[1.0,0.8], [0.8,1.0]]
cls1 = np.random.multivariate_normal([-2,2], cov, int(N/K))
cls2 = np.random.multivariate_normal([0,0], cov, int(N/K))
cls3 = np.random.multivariate_normal([2,-2], cov, int(N/K))
feature_phi = np.vstack((cls1,cls2,cls3))


#Creating Label Matrix T
Teacher_mat_T = np.vstack(([[1,0,0]]*cls1.shape[0],[[0,1,0]]*cls2.shape[0],[[0,0,1]]*cls3.shape[0]))
#Test
PHI = make_design_matrix_trans(feature_phi).transpose()
W = np.zeros((D+1,K))
I = np.identity(K)

while True:
    Y = soft_max(PHI @ W)
    #Updating weight matrix
    W_NEW = np.zeros((D+1,K))
    for cls_num in range(K):
        W_NEW[:,cls_num] = W[:,cls_num] - (np.linalg.inv(diag_Hesse_matrix(PHI,Y,cls_num)) @ PHI.T @ (Y-Teacher_mat_T))[:,cls_num]
    if np.allclose(W, W_NEW,rtol=0.01): break
    W = W_NEW



#Plotting data
x = np.arange(-10,10,0.01)
y = np.arange(-10,10,0.01)
cls1_x, cls1_y = np.transpose(np.array(cls1))
plt.scatter(cls1_x,cls1_y,c='r',marker='x',label="class1",s=40,linewidth=1.5)
cls2_x, cls2_y = np.transpose(np.array(cls2))
plt.scatter(cls2_x, cls2_y , color='limegreen',marker='+',linewidth=1.5,s=60)
cls3_x, cls3_y = np.transpose(np.array(cls3))
plt.scatter(cls3_x, cls3_y ,facecolor="none", edgecolor="b",label="class2",s=50,linewidth=1.5)
#Plotting Decision bouncary
plt.plot(x,decision_boundary(x,W.T,0,1),color='black',linewidth=0.8)
plt.plot(x,decision_boundary(x,W.T,1,2),color='black',linewidth=0.8)
#Drawing area
X,Y = np.meshgrid(x,y)
plt.fill_between(x,decision_boundary(x,W.T,0,1),decision_boundary(x,W.T,1,2),facecolor='lime',alpha=0.2)
plt.fill_between(x,10,decision_boundary(x,W.T,0,1),facecolor='lightcoral',alpha=0.2)
plt.fill_between(x,-10,decision_boundary(x,W.T,1,2),facecolor='cornflowerblue',alpha=0.1)


plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.show()
