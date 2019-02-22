
# coding: utf-8

# ## Trigger Word Detection

import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from td_utils import *


from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam


get_ipython().magic('matplotlib inline')


def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.
    """
    
    segment_start = np.random.randint(low=0, high=10000-segment_ms)   # Make sure segment doesn't run past the 10sec background 
    segment_end = segment_start + segment_ms - 1
    
    return (segment_start, segment_end)


def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.
    """
    segment_start, segment_end = segment_time
  	overlap = False
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True
    return overlap

# GRADED FUNCTION: insert_audio_clip

def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the 
    audio segment does not overlap with existing segments.
    """
    
    segment_ms = len(audio_clip)
    
    segment_time = get_random_time_segment(segment_ms)
    
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    previous_segments.append(segment_time)
 
    new_background = background.overlay(audio_clip, position = segment_time[0])
    return new_background, segment_time


def insert_ones(y, segment_end_ms):
   	segment_end_y = int(segment_end_ms * Ty / 10000.0)
    
    for i in range(segment_end_y + 1, segment_end_y + 51):
        if i < Ty:
            y[0, i] = 1
    
    return y


def create_training_example(background, activates, negatives):
    """
    Creates a training example with a given background, activates, and negatives.
    """
    # Set the random seed
    np.random.seed(18)
    
    # Make background quieter
    background = background - 20
    y = np.zeros((1, Ty))

    previous_segments = []
    
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    
    for random_activate in random_activates:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(y, segment_end_ms=segment_end)
    
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    for random_negative in random_negatives:
        # Insert the audio clip on the background 
        background, _ = insert_audio_clip(background, random_negative, previous_segments)
   
    background = match_target_amplitude(background, -20.0)

    file_handle = background.export("train" + ".wav", format="wav")
    print("File (train.wav) was saved in your directory.")
    
    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    x = graph_spectrogram("train.wav")
    
    return x, y


def model(input_shape):
    """
    Function creating the model's graph in Keras.
    """
    X_input = Input(shape = input_shape)
    
    # Step 1: CONV layer 
    X = Conv1D(196, kernel_size=15, strides=4)(X_input)                                 # CONV1D
    X = BatchNormalization()(X)                                 # Batch normalization
    X = Activation('relu')(X)                                 # ReLu activation
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)

    # Step 2: First GRU Layer 
    X = GRU(units = 128, return_sequences = True)(X) # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                                 # Batch normalization
    
    # Step 3: Second GRU Layer 
    X = GRU(units = 128, return_sequences = True)(X)   # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                                  # Batch normalization
    X = Dropout(0.8)(X)                                  # dropout (use 0.8)
    
    # Step 4: Time-distributed dense layer
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) # time distributed  (sigmoid)


    model = Model(inputs = X_input, outputs = X)
    
    return model  


def detect_triggerword(filename):
    plt.subplot(2, 1, 1)

    x = graph_spectrogram(filename)
    # the spectogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    
    plt.subplot(2, 1, 2)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()
    return predictions


def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    # Step 1: Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0
    # Step 2: Loop over the output steps in the y
    for i in range(Ty):
        # Step 3: Increment consecutive output steps
        consecutive_timesteps += 1
        # Step 4: If prediction is higher than the threshold and more than 75 consecutive output steps have passed
        if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
            # Step 5: Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds)*1000)
            # Step 6: Reset consecutive output steps to 0
            consecutive_timesteps = 0
        
    audio_clip.export("chime_output.wav", format='wav')



# Preprocess the audio to the correct format
def preprocess_audio(filename):
    # Trim or pad audio segment to 10000ms
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(filename)[:10000]
    segment = padding.overlay(segment)
    # Set frame rate to 44100
    segment = segment.set_frame_rate(44100)
    # Export as wav
    segment.export(filename, format='wav')


if __name__ == '__main__':
	x = graph_spectrogram("audio_examples/example_train.wav")
	_, data = wavfile.read("audio_examples/example_train.wav")
	print("Time steps in audio recording before spectrogram", data[:,0].shape)
	print("Time steps in input after spectrogram", x.shape)
	Tx = 5511 # The number of time steps input to the model from the spectrogram
	n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram

	Ty = 1375 # The number of time steps in the output of our model

	# Load audio segments using pydub 
	activates, negatives, backgrounds = load_raw_audio()

	print("background len: " + str(len(backgrounds[0])))    # Should be 10,000, since it is a 10 sec clip
	print("activate[0] len: " + str(len(activates[0])))     # Maybe around 1000, since an "activate" audio clip is usually around 1 sec (but varies a lot)
	print("activate[1] len: " + str(len(activates[1])))     # Different "activate" clips can have different lengths 

	overlap1 = is_overlapping((950, 1430), [(2000, 2550), (260, 949)])
	overlap2 = is_overlapping((2305, 2950), [(824, 1532), (1900, 2305), (3424, 3656)])
	print("Overlap 1 = ", overlap1)
	print("Overlap 2 = ", overlap2)

	x, y = create_training_example(backgrounds[0], activates, negatives)
	plt.plot(y[0])
	X = np.load("./XY_train/X.npy")
	Y = np.load("./XY_train/Y.npy")
	X_dev = np.load("./XY_dev/X_dev.npy")
	Y_dev = np.load("./XY_dev/Y_dev.npy")
	model = model(input_shape = (Tx, n_freq))
	model.summary()

	model = load_model('./models/tr_model.h5')


	opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

	model.fit(X, Y, batch_size = 5, epochs=1)

	loss, acc = model.evaluate(X_dev, Y_dev)
	print("Dev set accuracy = ", acc)
	chime_file = "audio_examples/chime.wav"

	filename = "./raw_data/dev/1.wav"
	prediction = detect_triggerword(filename)
	chime_on_activate(filename, prediction, 0.5)

	filename  = "./raw_data/dev/2.wav"
	prediction = detect_triggerword(filename)
	chime_on_activate(filename, prediction, 0.5)

	my_filename = "audio_examples/my_audio.wav"
	preprocess_audio(my_filename)
	chime_threshold = 0.5
	prediction = detect_triggerword(my_filename)
	chime_on_activate(my_filename, prediction, chime_threshold)
	#IPython.display.Audio("./chime_output.wav")
