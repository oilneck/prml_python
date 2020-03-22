from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt



# input data
read_img = cv2.imread("./image_data/my_data/8.png")
# get the center position
height,width,channel = read_img.shape[:3]
w_center = width//2
h_center = height//2
# trimming
trim_img = read_img[h_center-70:h_center+70, w_center-70:w_center+70]

# BGR -> gray scale
gray = cv2.cvtColor(trim_img, cv2.COLOR_BGR2GRAY)
# gray scale -> binary translation
_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
# binary -> black & white inversion
th = cv2.bitwise_not(th)
# blur processing
th = cv2.GaussianBlur(th,(9,9),0)
# save output data
cv2.imwrite("./image_data/binary_data/test.jpg", th)

# reading output data
test_img = cv2.imread("./image_data/binary_data/test.jpg")
image = []
for n in range(3):
    # resize from (140,140) to (28,28) in RGB space
    image.append(cv2.resize(test_img[:,:,n],(28, 28), cv2.INTER_CUBIC))
# nomalization
Xt = np.array(image)/255

'''generate prediction data'''
model = load_model("./TEST/model_data/cnn_MNIST.h5")
result = model.predict_classes(Xt)

#----showing figure------
fig = plt.figure(figsize=(7,3))
ax = fig.add_subplot(111)
# BGR -> RGB translation
im_color = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
plt.imshow(im_color)
ax_pos = ax.get_position()
fig.text(ax_pos.x1 - 0.15, ax_pos.y1 - 0.15, "prediction",fontsize=20)
fig.text(ax_pos.x1 - 0.1, ax_pos.y1 - 0.55, "{0}".format(result[0]),fontsize=60,color='r')
plt.show()
