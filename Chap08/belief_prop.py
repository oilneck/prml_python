import numpy as np
import matplotlib.pyplot as plt
from deepL_module.datasets.mnist import load_mnist
from graphx import *
from copy import deepcopy

def addNoise(image_, num=50):
    img = np.copy(image_).ravel()
    idx = np.random.choice(img.size, size=num)
    img[idx] = 1 - img[idx]
    return img.reshape(*image_.shape)

(X, _), (_, _) = load_mnist(normalize=True)
image_data = (X[7] > .5).astype(int).reshape(28,28)
noisy_image = addNoise(image_data)


def generateBeliefNetwork(image):
    network = MarkovRandomField()
    height, width = image.shape

    for i in range(height):
        for j in range(width):
            nodeID = width * i + j
            node = Node(nodeID)
            network.add_node(nodeID, node)

    dy = [-1, 0, 0, 1]
    dx = [0, -1, 1, 0]

    for i in range(height):
        for j in range(width):
            node = network.get_node(width * i + j)

            for k in range(4):
                if i + dy[k] >= 0 and i + dy[k] < height and j + dx[k] >= 0 and j + dx[k] < width:
                    neighbor = network.get_node(width * (i + dy[k]) + j + dx[k])
                    node.add_neighbor(neighbor)

    return network



network = generateBeliefNetwork(image_data)
for i in range(image_data.shape[0]):
    for j in range(image_data.shape[1]):
        node = network.get_node(image_data.shape[1] * i + j)
        node.likelihood(noisy_image[i,j])

network.message_passing(10)

output = np.zeros_like(noisy_image)

for i in range(output.shape[0]):
    for j in range(output.shape[1]):
        node = network.get_node(output.shape[1] * i + j)
        prob = node.prob
        if prob[1] > prob[0]:
            output[i,j] = 1

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
