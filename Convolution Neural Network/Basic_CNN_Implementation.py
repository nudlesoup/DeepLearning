#Author : Ameya Dhamanaskar
# coding: utf-8

# # Basic Convolutional Neural Networks


import numpy as np
import h5py
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.random.seed(1)
def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    """
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant')
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    """
    # Element-wise product between a_slice and W. Do not add the bias yet.
    s = np.multiply(a_slice_prev , W) 
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z+float(b)

    return Z

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    """
    # Retrieve dimensions from A_prev's shape 
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape 
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters" 
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Compute the dimensions of the CONV output volume using the formula given above.
    n_H = int((n_H_prev-f+2*pad)/stride) +1
    n_W = int((n_W_prev-f+2*pad)/stride) +1
    
    # Initialize the output volume Z with zeros.
    Z = np.zeros((m,n_H,n_W,n_C))
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev,pad)
    
    for i in range(m):                               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]                               # Select ith training example's padded activation
        for h in range(n_H):                           # loop over vertical axis of the output volume
            for w in range(n_W):                       # loop over horizontal axis of the output volume
                for c in range(n_C):                   # loop over channels (= #filters) of the output volume
                    
                    # Find the corners of the current "slice" 
                    vert_start = h*stride
                    vert_end = vert_start+f
                    horiz_start = w*stride
                    horiz_end = horiz_start+f
                    
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell).
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. 
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:,:,:,c], b[:,:,:,c])
                                        
    
    
    # Making sure your output shape is correct
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache

def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer
    """
    
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))              
    
   
    for i in range(m):                         # loop over the training examples
        for h in range(n_H):                     # loop on the vertical axis of the output volume
            for w in range(n_W):                 # loop on the horizontal axis of the output volume
                for c in range (n_C):            # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice" 
                    vert_start = h*stride
                    vert_end = vert_start+f
                    horiz_start = w*stride
                    horiz_end = horiz_start+f
                    
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. 
                    a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
                    
                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    
    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)
    
    # Making sure your output shape is correct
    assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache

def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function
    """
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache
    
    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape
    
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m,n_H_prev,n_W_prev,n_C_prev))                           
    dW = np.zeros((f,f,n_C_prev,n_C))
    db = np.zeros((1,1,1,n_C))

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev,pad)
    dA_prev_pad = zero_pad(dA_prev,pad)
    
    for i in range(m):                       # loop over the training examples
        
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        
        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice"
                    vert_start = h*stride
                    vert_end = vert_start+f
                    horiz_start = w*stride
                    horiz_end = horiz_start+f
                    
                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += np.multiply(W[:,:,:,c],dZ[i,h,w,c])
                    dW[:,:,:,c] += np.multiply(a_slice,dZ[i,h,w,c])
                    db[:,:,:,c] += np.sum(dZ[i,h,w,c])
                    
        # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    
    
    # Making sure your output shape is correct
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db


def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    """
    mask =(np.max(x)==x)
   
    return mask


def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape
    """
    
    (n_H, n_W) = shape
    
    # Compute the value to distribute on the matrix 
    average = np.sum(dz)/(n_H*n_W)
    
    # Create a matrix where every entry is the "average" value 
    a = np.full((n_H,n_W),average)
    ### END CODE HERE ###
    
    return a

def pool_backward(dA, cache, mode = "max"):
    """
    Implements the backward pass of the pooling layer
    """
    # Retrieve information from cache 
    (A_prev, hparameters) = cache
    
    # Retrieve hyperparameters from "hparameters" 
    stride = hparameters["stride"]
    f = hparameters["f"]
    
    # Retrieve dimensions from A_prev's shape and dA's shape 
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros((m,n_H_prev,n_W_prev,n_C_prev))
    
    for i in range(m):                       # loop over the training examples
        
        # select training example from A_prev (≈1 line)
        a_prev = A_prev[i]
        
        for h in range(n_H):                   # loop on the vertical axis
            for w in range(n_W):               # loop on the horizontal axis
                for c in range(n_C):           # loop over the channels (depth)
                    
                    # Find the corners of the current "slice" 
                    vert_start = h*stride
                    vert_end = vert_start+f
                    horiz_start = w*stride
                    horiz_end = horiz_start+f
                    
                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        
                        # Use the corners and "c" to define the current slice from a_prev 
                        a_prev_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,c]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) 
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask*dA[i,h,w,c]
                        
                    elif mode == "average":
                        
                        # Get the value a from dA 
                        da = dA[i,h,w,c]
                        # Define the shape of the filter as fxf 
                        shape = (f,f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. 
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da,shape)
                        
    
    # Making sure your output shape is correct
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev



if __name__ == '__main__':
    # Testing zero_pad
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    x_pad = zero_pad(x, 2)
    print ("x.shape =", x.shape)
    print ("x_pad.shape =", x_pad.shape)
    print ("x[1,1] =", x[1,1])
    print ("x_pad[1,1] =", x_pad[1,1])

    fig, axarr = plt.subplots(1, 2)
    axarr[0].set_title('x')
    axarr[0].imshow(x[0,:,:,0])
    axarr[1].set_title('x_pad')
    axarr[1].imshow(x_pad[0,:,:,0])
    a_slice_prev_1 = np.random.randn(4, 4, 3)
    W_1 = np.random.randn(4, 4, 3)
    b_1 = np.random.randn(1, 1, 1)
    Z_1 = conv_single_step(a_slice_prev_1, W_1, b_1)
    print("Z =", Z)
    # In
    A_prev_2 = np.random.randn(10,4,4,3)
    W_2 = np.random.randn(2,2,3,8)
    b_2 = np.random.randn(1,1,1,8)
    hparameters_2 = {"pad" : 2,
                   "stride": 2}
    Z_2, cache_conv_2 = conv_forward(A_prev_2, W_2, b_2, hparameters_2)
    print("Z's mean =", np.mean(Z))
    print("Z[3,2,1] =", Z[3,2,1])
    print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
    # In
    A_prev_3 = np.random.randn(2, 4, 4, 3)
    hparameters_3 = {"stride" : 2, "f": 3}
    A_3, cache_3 = pool_forward(A_prev_3, hparameters_3)
    print("mode = max")
    print("A =", A)
    print()
    A_3, cache_3 = pool_forward(A_prev_3, hparameters_3, mode = "average")
    print("mode = average")
    print("A =", A)
    # In[25]:
    dA, dW, db = conv_backward(Z_2, cache_conv_2)
    print("dA_mean =", np.mean(dA))
    print("dW_mean =", np.mean(dW))
    print("db_mean =", np.mean(db))
    # In[27]:
    x_4 = np.random.randn(2,3)
    mask_4 = create_mask_from_window(x_4)
    print('x = ', x_4)
    print("mask = ", mask_4)
   
    # In
    a = distribute_value(2, (2,2))
    print('distributed value =', a)
    A_prev_5 = np.random.randn(5, 5, 3, 2)
    hparameters_5  = {"stride" : 1, "f": 2}
    A_5 , cache_5  = pool_forward(A_prev_5 , hparameters_5 )
    dA_5  = np.random.randn(5, 4, 2, 2)
    dA_prev_5  = pool_backward(dA_5 , cache_5 , mode = "max")
    print("mode = max")
    print('mean of dA = ', np.mean(dA_5 ))
    print('dA_prev[1,1] = ', dA_prev_5 [1,1])  
    print()
    dA_prev_5  = pool_backward(dA_5 , cache_5 , mode = "average")
    print("mode = average")
    print('mean of dA = ', np.mean(dA_5 ))
    print('dA_prev[1,1] = ', dA_prev[1,1])
