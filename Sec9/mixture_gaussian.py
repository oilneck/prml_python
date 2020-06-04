import numpy as np
import matplotlib.pyplot as plt



covs = np.array([[[1,0],[0,1]],[[1,1],[-1,1]],[[1,0],[0,1]]])
param = np.array([1,2,3])
means = np.array([[0,0],[0,3],[-1,-1]])

def gauss(X):
    dev = X[None,:,:]-means[:,None,:]
    vect = np.einsum('kij, knj -> kni', np.linalg.inv(covs), dev)
    gauss = np.exp(-0.5 * np.sum(dev * vect, axis=-1)).T
    gauss /= np.sqrt(2 * np.pi * np.linalg.det(covs))
    return gauss

def gamma(X):
    nume = gauss(X) * param
    nume /= nume.sum(axis=-1)[:,None]
    return nume

def update_params(X):
    global covs,means
    resp = gamma(X)
    N_k = np.sum(resp,axis=0)
    param = N_k / len(X)
    means = resp.T @ X / N_k[:,None]
    diff = X[None,:,:]-means[:,None,:]
    diff_ = diff*gamma(X)[None,:,:].transpose(2,1,0)
    covs =  diff.transpose(0,2,1) @ diff_
    covs /= N_k[:,None,None]
    param = N_k / len(X)

def fit(X:np.ndarray, n_iter:int=100):
    for _ in range(n_iter):
        old_params = np.hstack((param.ravel(),covs.ravel(),means.ravel())).copy()
        update_params(X_train)
        if np.allclose(old_params, np.hstack((param.ravel(),covs.ravel(),means.ravel()))):
            break

def predict_proba(X):
    g = param * gauss(X)
    return np.sum(g, axis=-1)

def classify(X):
    joint_prob = param * gauss(X)
    return np.argmax(joint_prob, axis=1)

def create_toy_data():
    x1 = np.random.normal(size=(100, 2))
    x1 += np.array([-5, -5])
    x2 = np.random.normal(size=(100, 2))
    x2 += np.array([5, -5])
    x3 = np.random.normal(size=(100, 2))
    x3 += np.array([0, 5])
    return np.vstack((x1, x2, x3))


X_train =create_toy_data()
fit(X_train)
labels = classify(X_train)

x_test, y_test = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
X_test = np.array([x_test, y_test]).reshape(2, -1).transpose()
probs = predict_proba(X_test)
Probs = probs.reshape(100, 100)
colors = ["red", "blue", "green"]
plt.scatter(X_train[:, 0], X_train[:, 1], c=[colors[int(label)] for label in labels])
plt.contour(x_test, y_test, Probs)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()
