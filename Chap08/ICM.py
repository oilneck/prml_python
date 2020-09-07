import numpy as np
import matplotlib.pyplot as plt
from deepL_module.datasets.mnist import load_mnist
import itertools


def addNoise(image, num=50):
    img = np.copy(image).ravel()
    idx = np.random.choice(img.size, size=num)
    img[idx] *= -1
    return img.reshape(*image.shape)

def E(loc):
    neighbor = np.sum([output[tuple(loc + x)] for x in comb])
    energy = h * output.sum() - eta * (noisy_image * output).sum()
    energy -= beta * neighbor * output[loc]
    return energy

def ICM(loc):
    global output
    output[loc] = 1
    posi = E(loc)
    output[loc] = -1
    nega = E(loc)
    if posi < nega:
        output[loc] = 1
    else:
        output[loc] = -1

#1 Preparing image data
(data, _), _ = load_mnist(normalize=True)
origin_image = (data[7] > .5).astype(int).reshape(28, 28)
origin_image = origin_image * 2 - 1
noisy_image = addNoise(origin_image)
output = noisy_image.copy()
x_len, y_len = origin_image.shape

comb = set(itertools.product(range(-1, 2), range(-1, 2)))
comb = np.array(list(map(np.array, comb.difference((0,0)))))
h = 0.2
beta = .5
eta = 2

for _ in range(10):
    for loc in itertools.product(range(1, x_len-1), range(1, y_len-1)):
        ICM(loc)


# display images
images = {'origin':origin_image, 'noised':noisy_image, 'denoised':output}
for n, (text, disp) in enumerate(images.items()):
    ax = plt.subplot(1, 3, n+1)
    ax.imshow(disp, cmap='gray')
    ax.axis("off")
    plt.title(text, fontsize=20)
plt.tight_layout()
plt.show()
