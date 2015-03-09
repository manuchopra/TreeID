import numpy as np
import matplotlib.pyplot as plt

def init_two_layer_model(input_size, hidden_size, output_size):
  """
  Initialize the weights and biases for a two-layer fully connected neural
  network. The net has an input dimension of D, a hidden layer dimension of H,
  and performs classification over C classes. Weights are initialized to small
  random values and biases are initialized to zero.

  Inputs:
  - input_size: The dimension D of the input data
  - hidden_size: The number of neurons H in the hidden layer
  - ouput_size: The number of classes C

  Returns:
  A dictionary mapping parameter names to arrays of parameter values. It has
  the following keys:
  - W1: First layer weights; has shape (D, H)
  - b1: First layer biases; has shape (H,)
  - W2: Second layer weights; has shape (H, C)
  - b2: Second layer biases; has shape (C,)
  """
  # initialize a model
  model = {}
  model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model

def two_layer_net(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradients for a two layer fully connected neural network.
  The net has an input dimension of D, a hidden layer dimension of H, and
  performs classification over C classes. We use a softmax loss function and L2
  regularization the the weight matrices. The two layer net should use a ReLU
  nonlinearity after the first affine layer.

  Inputs:
  - X: Input data of shape (N, D). Each X[i] is a training sample.
  - model: Dictionary mapping parameter names to arrays of parameter values.
    It should contain the following:
    - W1: First layer weights; has shape (D, H)
    - b1: First layer biases; has shape (H,)
    - W2: Second layer weights; has shape (H, C)
    - b2: Second layer biases; has shape (C,)
  - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
  - reg: Regularization strength.

  Returns:
  If y not is passed, return a matrix scores of shape (N, C) where scores[i, c]
  is the score for class c on input X[i].

  If y is not passed, instead return a tuple of:
  - loss: Loss (data loss and regularization loss) for this batch of training
    samples.
  - grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function. This should have the same keys as model.
  """

  # unpack variables from the model dictionary
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, D = X.shape
  C = W2.shape[1]

  # compute the forward pass
  scores = None
  #############################################################################
  # TODO: Perform the forward pass, computing the class scores for the input. #
  # Store the result in the scores variable, which should be an array of      #
  # shape (N, C).                                                             #
  #############################################################################
  #############################################################################
  y1 = np.dot(X, W1) #h1
  z1 = np.maximum(y1 + b1,0) #h2 - apply ReLU
  scores = z1.dot(W2) + b2
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  
  # If the targets are not given then jump out, we're done
  if y is None:
    return scores

  # compute the loss
  loss = 0
  #############################################################################
  # TODO: Finish the forward pass, and compute the loss. This should include  #
  # both the data loss and L2 regularization for W1 and W2. Store the result  #
  # in the variable loss, which should be a scalar. Use the Softmax           #
  # classifier loss. So that your results match ours, multiply the            #
  # regularization loss by 0.5                                                #
  #############################################################################
  correct_scores = scores[range(N), y]
  logSum = np.log(np.sum(np.exp(scores), 1))
  loss_i = -correct_scores + logSum
  loss = np.average(loss_i)
  loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2)) #regularization

  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  # compute the gradients
  grads = {}
  #############################################################################
  # TODO: Compute the backward pass, computing the derivatives of the weights #
  # and biases. Store the results in the grads dictionary. For example,       #
  # grads['W1'] should store the gradient on W1, and be a matrix of same size #
  #############################################################################

  dW1 = np.zeros(W1.shape)
  dW2 = np.zeros(W2.shape)
  db1 = np.zeros(b1.shape)
  db2 = np.zeros(b2.shape)

  Y = np.zeros((y.shape[0],C))
  Y[range(N), y] = -1 #sets all Y[0,y], Y[1,y] ... Y[N,y] = -1
  
  expScore = np.exp(scores)
  margin = expScore.T / np.sum(expScore, axis=1)
  dW2 += (margin.dot(z1) / N).T
  dW2 += z1.T.dot(Y) / N #-1
  grads['W2'] = dW2 + reg * W2

  db2 += np.sum(margin, 1) / N
  db2 += np.sum(Y, 0) / N #-1
  grads['b2'] = db2

  temp1 = np.dot(W2,margin) - W2[:,y]
  temp1 = temp1 * (z1 > 0).T
  dW1 += (temp1.dot(X)).T / N
  grads['W1'] = dW1 + reg * W1

  db1 = np.sum(temp1, 1) / N
  grads['b1'] = db1

  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  return loss, grads

