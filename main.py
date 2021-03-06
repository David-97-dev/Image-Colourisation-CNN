import os
import random
from math import ceil
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import InputLayer, UpSampling2D, DepthwiseConv2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

# Loading in files and establishing route folder
os.path.join("D:", "Msc Computer Science", "Python", "Colournet", "images")

folder = 'D:\\Msc Computer Science\\Python\\Colournet\\images\\Train'

filenames = random.sample(os.listdir(folder),
                          10)  # replace with path of training images. #number represents sample size. Increase for greater accuracy

rootdir = os.getcwd()

dir = 'D:\\Msc Computer Science\\Python\\Colournet\\images\\Train'

lspace = []
abspace = []

# converting each image to greyscale using cv2. ##Then every image is converted to lab colour space.
# They are split into 2 arrays, lspace for light intensity, and ab space where a and b are the position iun the axis between
# red-green and blue-yellow ranges , respectively

for file in filenames:
    rgb = io.imread((os.path.join(dir, file)))
    lab_image = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)  # convert colors space from RGB to LAB
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    lspace.append(l_channel)
    replot_lab = np.zeros((256, 256, 2))
    replot_lab[:, :, 0] = a_channel
    replot_lab[:, :, 1] = b_channel
    abspace.append(replot_lab)
    transfer = cv2.merge([l_channel, a_channel, b_channel])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

lspace = np.asarray(lspace)  # convert to array
abspace = np.asarray(abspace)  # convert to array

np.save("lspace100.npy", lspace)
np.save("abspace100.npy", abspace)

X = np.load("lspace100.npy")
Y = np.load("abspace100.npy")

print(X.shape)

# Structuring model. Using CNN+VGG-16
model6 = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
model = Sequential()
model.add(InputLayer(input_shape=(X.shape[1], X.shape[2], 1)))
model.add(layers.Dense(units=3))
model.add(Model(inputs=model6.inputs, outputs=model6.
                layers[-10].output))
model.add(UpSampling2D((2, 2)))
model.add(UpSampling2D((2, 2)))
model.add(DepthwiseConv2D(32, (2, 2), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(DepthwiseConv2D(32, (2, 2), activation='tanh', padding='same'))
model.add(layers.ReLU(0.3))
model.add(layers.Dropout(0.4))
model.add(UpSampling2D((2, 2)))
model.add(UpSampling2D((2, 2)))
model.add(DepthwiseConv2D(2, (2, 2), activation='tanh', padding='same'))
model.add(layers.ReLU(0.3))
model.add(layers.Dropout(0.2))
model.add(UpSampling2D((2, 2)))
model.add(layers.ReLU(0.3))
model.add(layers.Dropout(0.2))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(layers.Dense(units=2))
print(model.summary())


# creating optimiser


def adam_optimizer():
    return Adam(lr=0.001, beta_1=0.99, beta_2=0.999)


model.compile(loss='mape', optimizer=adam_optimizer())

# Data prep
print('Preping learning data')
X = ((X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)))
X = (X - 255) / 255
Y = (Y - 255) / 255
trainsize = ceil(0.8 * X.shape[0])
testsize = ceil(0.2 * X.shape[0]) + 1

train_inp = X[:trainsize, ]
test_inp = X[testsize:, ]
train_out = Y[:trainsize, ]
test_out = Y[testsize:, ]

# train model

print(train_inp.shape)
print(test_inp.shape)
print('Running model fit')

model.fit(x=train_inp, y=train_out, batch_size=1, epochs=1)
model.save('saved_model/my_model')
