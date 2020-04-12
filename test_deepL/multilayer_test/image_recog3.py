import numpy as np
import matplotlib.pyplot as plt
from deepL_module.datasets.mnist import load_mnist
from deepL_module.base import *
from skimage import color
from skimage.util import invert
from skimage.transform import rescale, resize, downscale_local_mean


''' #1 load model data '''
path_r = './../../prml/deepL_module/datasets/model_data/im_model.pkl'
model = load_model(path_r)


''' #2 load image data '''
path_img = './../../test_keras/image_recognition/image_data/my_data/test_image.png'
img = plt.imread(path_img)
image = color.rgb2gray(img)


''' #3 processing image '''
image_resized = resize(image, (image.shape[0] / 5, image.shape[1] / 5),
                       anti_aliasing=True)
data = invert(image_resized)



''' #4 prediction data'''
Xt = data.ravel().reshape(1,784)
prob = model.predict(Xt)
pred = np.argmax(prob)


''' #5 showing image '''
fig = plt.figure(figsize=(11,4))
ax = fig.add_subplot(111)
ax.imshow(invert(image), cmap='gray')
plt.tick_params(labelbottom = False,
                labelleft = False,
                bottom = False,
                left = False)


# --- prediction ---
pos = ax.get_position()
pos_y = 0.5 * (pos.y1 - pos.y0)
fig.text(0.75, pos_y, str(pred), fontsize=60, color='r')
fig.text(0.71,0.65, "prediction",
        fontsize=20,
        transform=fig.transFigure,
        color='r')
plt.show()
