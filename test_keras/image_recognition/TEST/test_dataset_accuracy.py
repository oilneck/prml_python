from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras import losses
from tensorflow.keras.optimizers import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

(X_train, train_t), (X_test, test_t) = mnist.load_data()
X_train = X_train.reshape(60000, 784)/255
X_test = X_test.reshape(10000, 784)/255
# Binary translation
train_t = to_categorical(train_t)
test_t =  to_categorical(test_t)


'''Neural network design'''
model = Sequential()
model.add(Dense(512, input_dim=784,activation='relu'))
model.add(Dense(10,activation='softmax'))
optimize_routine = SGD(lr=0.1)
model.compile(
            loss="categorical_crossentropy",
            optimizer=optimize_routine,
            metrics=["accuracy"]
            )

# learning train datasets
hist = model.fit(
                X_train, train_t,
                batch_size=200,
                verbose=0,
                epochs=10,
                validation_split=0.1
                )

score = model.evaluate(X_test, test_t, verbose=1)
print("Accuracy rate = {0}".format(score[1]))

model.save("./model_data/MNIST.h5")
#model = load_model("MNIST.h5")
