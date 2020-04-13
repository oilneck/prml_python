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
path_img = './../../test_keras/image_recognition/image_data/my_data/_image.png'
read_img = plt.imread(path_img)
gray_img = color.rgb2gray(read_img)

# resize image
width, height = gray_img.shape
cent_w, cent_h = width // 2, height // 2
cut_size = 300 # trimming size [pix]
margin = cut_size // 2
image = gray_img[cent_h - margin : cent_h + margin,
                 cent_w - margin : cent_w + margin]


''' #3 processing image '''
image_resized = rescale(image, 1. / (cut_size / 28.),
                        anti_aliasing = True,
                        multichannel = False,
                        anti_aliasing_sigma = 1.5)
data = invert(image_resized)



''' #4 prediction data'''
Xt = data.ravel().reshape(1,784)
prob = model.predict(Xt)
pred = np.argmax(prob)


''' #5 showing image '''
fig = plt.figure(figsize=(11,4))
ax = fig.add_subplot(111)
ax.imshow(data, cmap='gray')
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
