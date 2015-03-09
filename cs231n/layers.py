import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  out = np.dot(x.reshape(x.shape[0], np.prod(x.shape[1:])),w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  
  dx = np.dot(dout,w.T).reshape(x.shape)
  di = np.prod(x.shape[1:]) #weird - tried using x.shape[1] - didn't work.
  dw = np.dot(x.reshape(x.shape[0], di).T,dout) #product of all d (i)
  db = np.sum(dout, axis=0)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  
  out = np.maximum(0,x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout.copy()
  dx[x<0] = 0

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data.
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  
  #First step - Assign values to inputs
  N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
  F, C, HH,WW = w.shape[0], w.shape[1], w.shape[2], w.shape[3]
  pad = conv_param['pad']
  stride = conv_param['stride']

  #Second step - Create the Hloop and Wloop variables
  
  Hloop = 1 + (H + 2 * pad - HH) / stride
  Wloop = 1 + (W + 2 * pad - WW) / stride
  
  out = np.zeros((N, F, Hloop, Wloop))
  
  pad_x = np.pad(x, ((0,0), (0,0), (pad, pad), (pad, pad)), 'constant',constant_values=0) #pad every image

  for i in xrange(N): #loop over every image
      x1 = pad_x[i] #get current image
      for f in xrange(F):
            temp = np.zeros((Hloop,Wloop))
            for hx in xrange(Hloop):
                for wx in xrange(Wloop):
                    hx_start, hx_final = hx * stride, hx * stride + HH
                    wx_start, wx_final = wx * stride, wx * stride + HH
                    
                    curr = x1[:, hx_start:hx_final, wx_start:wx_final]
                    
                    temp[hx, wx] = np.sum(np.multiply(curr, w[f])) + b[f]
            out[i,f] = temp

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  (x, w, b, conv_param) = cache

  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
 
#First step - Assign values to inputs
  N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
  F, C, HH,WW = w.shape[0], w.shape[1], w.shape[2], w.shape[3]
  pad = conv_param['pad']
  stride = conv_param['stride']
  pad_x = np.pad(x, ((0,0), (0,0), (pad, pad), (pad, pad)), 'constant',constant_values=0) #pad every image

#Second step - Create the Hloop and Wloop variables

  Hloop = 1 + (H + 2 * pad - HH) / stride
  Wloop = 1 + (W + 2 * pad - WW) / stride

  dw = np.zeros(w.shape)
  db = np.zeros(b.shape)
  ipad = np.zeros_like(pad_x)

  for hx in xrange(Hloop):
      for wx in range(Wloop):
            hx_start, hx_final = hx * stride, hx * stride + HH
            wx_start, wx_final = wx * stride, wx * stride + HH
            
            curr = pad_x[:,:,hx_start:hx_final, wx_start:wx_final]
            
            w1 = np.reshape(w,(-1,F))
            dout2 = dout[:,:,hx,wx]
            
            x_add, w_add, b_add = affine_backward(dout2,(curr,w1,b))
            
            ipad[:,:,hx_start:hx_final, wx_start:wx_final] += dout2.dot( w.reshape(w.shape[0], w.shape[1] * w.shape[2] * w.shape[3])).reshape((x.shape[0], x.shape[1], w.shape[2], w.shape[3]))
            
            dw += np.reshape(w_add.T, (w.shape))
            
            for i in xrange(N):
                for f in xrange(F):
                    db[f] += np.sum(dout[i, f, hx, wx],axis=0) #I initially wanted to do this in just 2 loops but due to a bug, I went back to 4 loop structure here.

  dx = ipad[:,:, pad:-pad, pad:-pad] #remove the padding
  return dx, dw, db

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
  height = pool_param['pool_height']
  width = pool_param['pool_width']
  stride = pool_param['stride']
  
  Hloop = 1 + (H - height) / stride
  Wloop = 1 + (W - width) / stride
  
  out = np.zeros((N, C, Hloop, Wloop))
  
  for i in xrange(N):
      for c in xrange(C):
          x1 = x[i, c] #current image
          for hx in xrange(Hloop):
              for wx in xrange(Wloop):
                  hx_start, hx_final = hx * stride, hx * stride + height
                  wx_start, wx_final = wx * stride, wx * stride + width
                  
                  curr = x1[hx_start:hx_final, wx_start:wx_final]
                  out[i, c, hx, wx] = np.max(curr) #take maximum for every single picture
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x = cache[0]
  N, C, H, W =  x.shape[0], x.shape[1], x.shape[2], x.shape[3]
  pool_param = cache[1]
  stride = pool_param['stride']
  height = pool_param['pool_height']
  width = pool_param['pool_width']

  dx = np.zeros(x.shape)
  Hloop = 1 + (H - height) / stride
  Wloop = 1 + (W - width) / stride

  for i in xrange(N):
     for c in xrange(C):
        for hx in xrange(Hloop):
            for wx in xrange(Wloop):
                hx_start, hx_final = hx * stride, hx * stride + height
                wx_start, wx_final = wx * stride, wx * stride + width
                    
                curr =  x[:, :, hx_start:hx_final,wx_start:wx_final]
                index = np.unravel_index(np.argmax(curr[i, c]), (height, width)) #Credit - Piazza post for suggesting use of unravel index
                
                dx[i, c, hx_start + index[0], wx_start + index[1]] += dout[i, c, hx, wx]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx
def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

