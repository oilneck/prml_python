import numpy as np
import matplotlib.pyplot as plt
from deepL_module.datasets.mnist import load_mnist
from graphx import *
import itertools


def addNoise(image, num=50):
    img = np.copy(image).ravel()
    idx = np.random.choice(img.size, size=num)
    img[idx] = np.array(~img[idx].astype(bool), dtype=int)
    return img.reshape(*image.shape)


def generateMarkovNetwork(image):

    mrf = MarkovRandomField()
    for nodeID, _ in enumerate(itertools.product(range(x_len), range(y_len))):
            node = Node()
            mrf.add_node(nodeID, node)

    for n, (x,y) in enumerate(itertools.product(range(x_len), range(y_len))):
        node = mrf.get_node(n)
        for dx, dy in itertools.permutations(range(-1, 2), 2):
            try:
                neighbor = mrf.get_node(y_len * (x + dx) + y + dy)
                node.add_neighbor(neighbor)
            except Exception:
                pass

    return mrf


def removalNoise(noised_image):

    output = np.zeros_like(noised_image)
    for n, loc in enumerate(itertools.product(range(x_len), range(y_len))):
        node = network.get_node(n)
        output[loc] = np.argmax(node.prob)

    return output


#1 Preparing image data
(data, _), _ = load_mnist(normalize=True)
origin_image = (data[7] > .5).astype(int).reshape(28, 28)
noisy_image = addNoise(origin_image)
x_len, y_len = origin_image.shape

#2 constructing Markov Random field
network = generateMarkovNetwork(origin_image)

#3 setting obeserved value
for n, loc in enumerate(itertools.product(range(x_len), range(y_len))):
    node = network.get_node(n)
    node.likelihood(noisy_image[loc])


''' #4 sum-product algorithm '''
network.message_passing(n_iter=10)


#5 image de-noising
output = removalNoise(noisy_image)

#6 display images
images = {'origin':origin_image, 'noised':noisy_image, 'denoised':output}
for n, (text, disp) in enumerate(images.items()):
    ax = plt.subplot(1, 3, n+1)
    ax.imshow(disp, cmap='gray')
    ax.axis("off")
    plt.title(text, fontsize=20)
plt.tight_layout()
plt.show()
