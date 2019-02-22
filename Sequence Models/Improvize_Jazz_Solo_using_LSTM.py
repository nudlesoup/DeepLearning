# Author : Ameya Dhamanaskar
# coding: utf-8

# # Improvise a Jazz Solo with an LSTM Network


from __future__ import print_function
import IPython
import sys
from music21 import *
import numpy as np
from grammar import *
from qa import *
from preprocess import * 
from music_utils import *
from data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K




def djmodel(Tx, n_a, n_values):
    """
    Implement the model
    """
    X = Input(shape=(Tx, n_values))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
 
    outputs = []
    
    # Step 2: Loop
    for t in range(Tx):
        x = Lambda(lambda x: X[:,t,:])(X)
        x = reshapor(x)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)
        
    # Step 3: Create model instance
    model = Model(inputs=[X, a0, c0], outputs=outputs)
    
    
    return model


def music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    """
    # Define the input of your model with a shape 
    x0 = Input(shape=(1, n_values))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    ### START CODE HERE ###
    # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
    outputs = []
    
    # Step 2: Loop over Ty and generate a value at every time step
    for t in range(Ty):
        
        # Step 2.A: Perform one step of LSTM_cell (≈1 line)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        
        # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
        out = densor(a)

        # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)
        outputs.append(out)
        
        # Step 2.D: Select the next value according to "out", and set "x" to be the one-hot representation of the
        #           selected value, which will be passed as the input to LSTM_cell on the next step. We have provided 
        #           the line of code you need to do this. 
        x = Lambda(one_hot)(out)
        
    # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    
    return inference_model


def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                       c_initializer = c_initializer):
    """
    Predicts the next value of values using the inference model.
    """
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    # Step 2: Convert "pred" into an np.array() of indices with the maximum probabilities
    indices = np.argmax(pred, axis=-1)
    # Step 3: Convert indices to one-hot vectors, the shape of the results should be (1, )
    results = to_categorical(indices, num_classes=78)
    
    return results, indices

if __name__ == '__main__':
    X, Y, n_values, indices_values = load_music_utils()
    print('shape of X:', X.shape)
    print('number of training examples:', X.shape[0])
    print('Tx (length of sequence):', X.shape[1])
    print('total # of unique values:', n_values)
    print('Shape of Y:', Y.shape)
    
    n_a = 64 

    reshapor = Reshape((1, 78))                        # Used in Step 2.B of djmodel(), below
    LSTM_cell = LSTM(n_a, return_state = True)         # Used in Step 2.C
    densor = Dense(n_values, activation='softmax')     # Used in Step 2.D

    model = djmodel(Tx = 30 , n_a = 64, n_values = 78)
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    m = 60
    a0 = np.zeros((m, n_a))
    c0 = np.zeros((m, n_a))

    model.fit([X, a0, c0], list(Y), epochs=100)
    inference_model = music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 50)

    x_initializer = np.zeros((1, 1, 78))
    a_initializer = np.zeros((1, n_a))
    c_initializer = np.zeros((1, n_a))

    results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
    print("np.argmax(results[12]) =", np.argmax(results[12]))
    print("np.argmax(results[17]) =", np.argmax(results[17]))
    print("list(indices[12:18]) =", list(indices[12:18]))


    out_stream = generate_music(inference_model)