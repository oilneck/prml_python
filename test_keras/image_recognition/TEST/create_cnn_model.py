from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, Reshape, MaxPooling2D, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

# Constructing model
model = Sequential()

model.add(Reshape((28,28,1), input_shape=(28,28)))
model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(16,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(784))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation("softmax"))

#-------------------------------------------------------
'''Load MNIST data'''

(X_train, train_t), (X_test, test_t) = mnist.load_data()

X_train = np.array(X_train)/255
X_test = np.array(X_test)/255

# binary translation
train_t = to_categorical(train_t)
test_t = to_categorical(test_t)

# compile
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# learning train dataset
hist = model.fit(X_train, train_t, batch_size=200, verbose=0,
                 epochs=10, validation_split=0.1)

# evaluation
score = model.evaluate(X_test, test_t, verbose=0)
print("Accuracy rate = {0}".format(score[1]))

# preserve model
model.save("cnn_MNIST.h5")
