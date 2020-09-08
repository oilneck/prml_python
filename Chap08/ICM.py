import numpy as np
import matplotlib.pyplot as plt
from deepL_module.datasets.mnist import load_mnist
import itertools


def addNoise(image, num=50):
    img = np.copy(image).ravel()
    idx = np.random.choice(img.size, size=num)
    img[idx] *= -1
    return img.reshape(*image.shape)


def get_energy(loc): # calc. Hamiltonian
    delta = list(itertools.product(range(-1, 2), range(-1, 2)))
    Del = np.sum([X_out[tuple(idx)] for idx in np.array(loc) + delta])
    E = h * X_out.sum() - eta * (X_in * X_out).sum() - beta * Del * X_out[loc]
    return E


def ICM(loc):
    global X_out
    E = []
    X_out[loc] = 1
    E.append(get_energy(loc))
    X_out[loc] = -1
    E.append(get_energy(loc))
    X_out[loc] = 2 * np.argmax(E) - 1


#1 Preparing image data
(data, _), _ = load_mnist(normalize=True)
origin = np.where(data[7] > 0.5, 1, -1).reshape(28,28)
X_in = addNoise(origin)
X_out = X_in.copy()

#2 Setting Hamiltonian params
h = 0.2
beta = .5
eta = 2


'''#3 ICM algorithm'''
for _ in range(10):
    for loc in itertools.product(range(1, 27), range(1, 27)):
        ICM(loc)


#4 display images
padding = np.pad(np.ones((26, 26)), (1, 1), 'constant')
images = {'origin':origin, 'noised':X_in, 'denoised':X_out * padding}
for n, (text, disp) in enumerate(images.items()):
    ax = plt.subplot(1, 3, n+1)
    ax.imshow(disp, cmap='gray')
    ax.axis("off")
    plt.title(text, fontsize=20)
plt.tight_layout()
plt.show()
