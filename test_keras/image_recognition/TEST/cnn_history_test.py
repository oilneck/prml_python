from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, Reshape, MaxPooling2D, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from IPython.display import SVG
import pydot_ng as pydot
import numpy as np
import matplotlib.pyplot as plt


model = load_model("./model_data/cnn_MNIST.h5")
import json
f = open('./model_data/cnn_MNIST_hist.json', 'r')
history = json.load(f)
f.close()


(X_train, train_t), (X_test, test_t) = mnist.load_data()

X_train = np.array(X_train)/255
X_test = np.array(X_test)/255

# binary translation
train_t = to_categorical(train_t)
test_t = to_categorical(test_t)

score = model.evaluate(X_test, test_t, verbose=0)
print("Accuracy rate = {0}".format(score[1]))
#plot_model(model,to_file='cnn_model.png')
#print("\n {}\n".format(model.summary()))


fig = plt.figure(figsize=(13,4))
ax = fig.add_subplot(1,2,1)
ax.plot(history['acc'],color='r',label='training')
ax.plot(history['val_acc'],color='lime',label='validation')
plt.legend(fontsize=15)
plt.xlabel('epoch',fontsize=15)
plt.title('model accuracy',fontsize=15)
plt.xlim(0,20-1)
ax = fig.add_subplot(1,2,2)
ax.plot(history['loss'],color='r',label='training')
ax.plot(history['val_loss'],color='lime',label='validation')
plt.legend(fontsize=15)
plt.title('model loss',fontsize=15)
plt.xlabel('epoch',fontsize=15)
plt.xlim(0,20-1)
plt.tight_layout()
plt.subplots_adjust(wspace=0.5)
plt.show()
