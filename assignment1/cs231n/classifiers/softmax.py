import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W) # D x C

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W) # C, 
    sc_exp = np.exp(scores)
    sc_exp_sum = np.sum(sc_exp)
    sc_exp_correct = sc_exp[y[i]]
    f_i = sc_exp_correct / sc_exp_sum
    loss += -np.log(f_i)
    
    temp = sc_exp # C,
    temp[y[i]] -= sc_exp_sum
    temp /= sc_exp_sum
    dW += np.dot(np.reshape(X[i], (-1, 1)), np.reshape(temp, (1, -1)))
       
  loss /= num_train
  loss += reg*np.sum(W*W)
  
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W) # D x C

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  
  scores = np.dot(X, W) # N x C
  sc_exp = np.exp(scores) # N x C
  sc_exp_sum = np.sum(sc_exp, axis=1) # N?

  f = sc_exp[range(num_train), y] / sc_exp_sum # N?
  loss = -np.sum(np.log(f))
  loss /= num_train
  loss += reg*np.sum(W*W)

  sc_exp[range(num_train), y] -= sc_exp_sum
  sc_exp = sc_exp.T # C X N
  sc_exp /= sc_exp_sum
  dW = np.dot(sc_exp, X).T # D x C
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

