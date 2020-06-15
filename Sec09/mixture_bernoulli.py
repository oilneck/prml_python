import numpy as np
import matplotlib.pyplot as plt
from deepL_module.datasets.mnist import load_mnist
from pd import *



def fetch_mnist(fetch_nums:list):
    (x, y), (_, _) = load_mnist(normalize=True)
    X = np.array([x[np.where(y == i)[0][:200]] for i in fetch_nums])
    X = (1 - X).reshape(200 * len(fetch_nums), 784) # binary inversion
    return (X > .5).astype(np.float)

# loading mnist data
X_train = fetch_mnist([2, 3, 4])


''' EM-estimation '''
prob_dist = MultivariateBernoulli(n_components=3)
prob_dist.fit(X_train)


# show means
plt.figure(figsize=(12, 5))
for i, mean in enumerate(prob_dist.means):
    plt.subplot(1, 3, i + 1)
    plt.imshow(mean.reshape(28, 28), cmap="gray")
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.tick_params(bottom=False, left=False)
plt.show()
