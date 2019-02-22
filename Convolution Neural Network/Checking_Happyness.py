# Author : Ameya Dhamanaskar
# coding: utf-8

# Checking Happyness in the Happy House


import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

get_ipython().magic('matplotlib inline')

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.

    """
    X_Input=Input(input_shape)
    X=ZeroPadding2D((3,3))(X_Input)
    X=Conv2D(32,(7,7),strides=(1,1),name='conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    # Create model.
    model = Model(inputs = X_Input, outputs = X, name='HappyModel')
    return model

if __name__ == '__main__':
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # Normalize image vectors
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
    # Reshape
    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T
    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))

    happyModel = HappyModel(X_train.shape[1:])
    happyModel.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    happyModel.fit(X_train,Y_train,epochs=15,batch_size=40)

    preds = happyModel.evaluate(X_test,Y_test)
    print()
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))

    img_path = 'images/my_image.jpg'
    img = image.load_img(img_path, target_size=(64, 64))
    imshow(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(happyModel.predict(x))

    happyModel.summary()
    plot_model(happyModel, to_file='HappyModel.png')
    SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))