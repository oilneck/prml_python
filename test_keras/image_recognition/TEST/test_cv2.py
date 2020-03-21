import cv2
import matplotlib.pyplot as plt

img = cv2.imread("./../image_data/SIDBA/Lenna.bmp")
image = cv2.rectangle(img, (50, 50), (100, 100), (255, 0, 0))
im_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
im_gray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                    cv2.COLOR_BGR2RGB)
#NOTICE : opencv = (B,G,R), matplotlib = (R,G,B)

fig = plt.figure(figsize=(7,3))
fig.add_subplot(1,2,1)
plt.imshow(im_color)
fig.add_subplot(1,2,2)
plt.imshow(im_gray)
plt.show()
