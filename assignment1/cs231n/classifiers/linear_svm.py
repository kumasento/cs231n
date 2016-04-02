import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  # for each training image in the dataset
  for i in xrange(num_train):
    # get the score vector
    scores = X[i].dot(W)
    # get the score of correct class
    correct_class_score = scores[y[i]]

    # for gradient
    num_classes_larger_than_margin = 0
    # iterate over other classes
    for j in xrange(num_classes):
      # skip correct class
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      # the max function
      if margin > 0:
        # update other than y[i]
        dW[:, j] += X[i]
        num_classes_larger_than_margin += 1
        loss += margin
    
    # update only the correct class
    dW[:, y[i]] += - X[i] * num_classes_larger_than_margin
      

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  correct_scores = np.choose(y, scores.T)
  margins = (scores.T-correct_scores).T + 1
  margins_mask = margins <= 0
  margins[margins_mask] = 0.0 

  losses = np.sum(margins, axis=1)
  loss = np.mean(losses)
  loss += 0.5 * reg * np.sum(W * W) - 1
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margins_mask = margins_mask.astype(int)
  margins_mask = 1 - margins_mask

  margins_mask[np.arange(num_train), y] = - np.sum(margins_mask, axis=1)
  
  # How to understand this magic:
  # Each row of margins_mask.T is a list of 'first class' coeffiencies for each
  # data row.
  # Then each row in the final result is np.sum(X_after[i], axis=0). 
  # X_after is X * m[:, i]
  dW = np.dot(margins_mask.T, X).T / num_train

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
