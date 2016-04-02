import numpy as np
from random import shuffle

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
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train, dim = X.shape
  num_class = W.shape[1]

  for i in xrange(num_train):
    Xi, yi = X[i], y[i]
    scores = Xi.dot(W)

    # Then run Softmax loss calculation, score is f in the formula
    scores -= np.max(scores) # numerical issue
    scores_sum = np.sum(np.exp(scores))
    score_yi = np.exp(scores[yi]) / scores_sum 
    loss_i = -np.log(score_yi)
    loss += loss_i

    # gradiant
    for j in xrange(num_class):
      prob = np.exp(scores[j])/scores_sum
      if j == yi:
        dW[:, j] += -Xi*(1-prob)
      else:
        dW[:, j] += Xi*prob

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_train
  dW += reg * W

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
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train, dim = X.shape
  num_class = W.shape[1]

  scores = X.dot(W) # scores for all train data.
  scores = (scores.T-np.max(scores, axis=1)).T
  exp_scores = np.exp(scores)
  scores = (exp_scores.T/np.sum(exp_scores, axis=1)).T
  losses = np.choose(y, scores.T)
  losses = -np.log(losses)
  loss = np.mean(losses) + 0.5 * reg * np.sum(W * W)

  """
  About this trick:
  To find matrix calculation in dW's calculation, we take one element (i,j)
  in dW.
  dW[i][j] = sum(score[t][j] * X[t][i])/num_train
  i is the index(0~dim-1), j is the class id, t is the index of train set
  It's very easy to find a matrix multiplication pattern in the formula.
  """
  mask = np.zeros((num_train, num_class))
  # remenber this trick:
  mask[range(num_train),y] = 1
  scores += -mask
  dW = X.T.dot(scores)
  dW /= num_train
  dW += reg * W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

