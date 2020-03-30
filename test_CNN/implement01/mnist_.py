import numpy as np
from cnn_module.datasets.mnist import load_mnist
import matplotlib.pyplot as plt
from PIL import Image

def img_show(img):
    Image.fromarray(np.uint8(img)).show()

(X_train, train_t), (X_test, test_t) = load_mnist(flatten = True, normalize = False)

img = X_train[0].reshape(28,28)
img_show(img)
