
import random
from math import ceil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import main
from skimage import io
import tensorflow as tf

model = tf.keras.models.load_model('saved_model/my_model')

print('Prepping test image')
# prepping test image
first_image_path = 'D:\Msc Computer Science\\Python\\Image Colourisation - Convolutional Neural Network\\firstimage.png'

first_lspace = []
first_abspace = []

im = io.imread(first_image_path)
lab_image = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
l_channel, a_channel, b_channel = cv2.split(lab_image)
first_lspace.append(l_channel)
replot_lab = np.zeros((256, 256, 2))
replot_lab[:, :, 1] = a_channel
replot_lab[:, :, 1] = b_channel
first_abspace.append(replot_lab)
first_image = cv2.merge([l_channel, a_channel, b_channel])
first_image = cv2.cvtColor(first_image.astype("uint8"), cv2.COLOR_LAB2BGR)

first_lspace = np.asarray(first_lspace)
first_abspace = np.asarray(first_abspace)

np.save("test_lspace100.npy", first_lspace)
np.save("test_abspace100.npy", first_abspace)

X_test = np.load("test_lspace100.npy")
Y_test = np.load("test_abspace100.npy")

plt.imsave('transfer_test.png', arr=first_image)

X_test = ((X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)))
X_test = (X_test - 255) / 255
Y_test = (Y_test - 255) / 255
trainsize_test = ceil(0.8 * X_test.shape[0])
testsize_test = ceil(0.2 * X_test.shape[0]) + 1

train_inp_test = X_test[:trainsize_test, ]
test_inp_test = X_test[testsize_test:, ]
# end of prep
print('My test shape' + str(test_inp_test.shape))

train_predictions = model.predict(train_inp_test)

test_predictions = model.predict(test_inp_test)

train_random = random.randint(1, trainsize_test)
test_random = random.randint(1, testsize_test)
check = np.interp(train_predictions, (train_predictions.min(), train_predictions.max()), (0, 255))
check1 = np.interp(test_predictions, (test_predictions.min(), test_predictions.max()), (0, 255))
#l_channel = test_inp[20] * 255
#a_channel = check1[20, :, :, 0]
#b_channel = check1[20, :, :, 1]

l_channel_test = test_inp_test[6] * 255
a_channel_test = check1[6, :, :, 0]
b_channel_test = check1[6, :, :, 1]
transfer = cv2.merge([l_channel_test, a_channel_test, b_channel_test])
transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

#view results

plt.imsave('transfer.png', arr=transfer)
plt.imshow(transfer)
