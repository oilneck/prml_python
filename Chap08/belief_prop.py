import numpy as np
import matplotlib.pyplot as plt
from deepL_module.datasets.mnist import load_mnist
from graphx import *
import itertools

def addNoise(image_, num=50):
    img = np.copy(image_).ravel()
    idx = np.random.choice(img.size, size=num)
    img[idx] = 1 - img[idx]
    return img.reshape(*image_.shape)

(data, _), (_, _) = load_mnist(normalize=True)
image_data = (data[7] > .5).astype(int).reshape(28,28)
noisy_image = addNoise(image_data)
X, Y = image_data.shape

def generateBeliefNetwork(image):
    network = MarkovRandomField()
    dy = [-1, 0, 0, 1]
    dx = [0, -1, 1, 0]

    for nodeID, (i,j) in enumerate(itertools.product(range(X), range(Y))):
            node = Node(nodeID)
            network.add_node(nodeID, node)


    for n, (i,j) in enumerate(itertools.product(range(X), range(Y))):
            node = network.get_node(n)

            for k in range(4):
                if i + dy[k] >= 0 and i + dy[k] < X and j + dx[k] >= 0 and j + dx[k] < Y:
                    neighbor = network.get_node(Y * (i + dy[k]) + j + dx[k])
                    node.add_neighbor(neighbor)

    return network



network = generateBeliefNetwork(image_data)
for n, (i,j) in enumerate(itertools.product(range(X), range(Y))):
    node = network.get_node(n)
    node.likelihood(noisy_image[i,j])

network.message_passing(n_iter=10)

output = np.zeros_like(noisy_image)

for n, (i,j) in enumerate(itertools.product(range(X), range(Y))):
    node = network.get_node(n)
    output[i,j] = np.argmax(node.prob)

fig = plt.figure(figsize=(11,4))
ax = fig.add_subplot(131)
ax.imshow(image_data, cmap='gray')
ax.axis("off")

ax = fig.add_subplot(132)
ax.imshow(noisy_image, cmap='gray')
ax.axis("off")

ax = fig.add_subplot(133)
ax.imshow(output, cmap='gray')
ax.axis("off")
plt.show()
