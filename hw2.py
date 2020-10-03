import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.special as sp
import time
from scipy.optimize import minimize
import data_generator as dg

# you can define/use whatever functions to implememt

########################################
# Part 1. cross entropy loss
########################################
def cross_entropy_softmax_loss(Wb, x, y, num_class, n, feat_dim):
    # implement your function here
    
    # total s value
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim))
    x=np.reshape(x.T, (-1, n))
    
    total_s = W@x + b

    # calculate loss value
    loss = 0
    for n_case in range(total_s.shape[1]):
      s_case_sum = 0
      s_case_one = 0
    
      for n_class in range(total_s.shape[0]):
        s_case_sum += np.exp(total_s[n_class][n_case])

        if y[n_case] == n_class:
          s_case_one = np.exp(total_s[n_class][n_case])
      
      loss += (-np.log(s_case_one/s_case_sum))

    # return cross entropy loss      
    avg_loss = loss/total_s.shape[1]
    
    return avg_loss

########################################
# Part 2. SVM loss calculation
########################################
def svm_loss(Wb, x, y, num_class, n, feat_dim):
    # implement your function here
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim))
    x=np.reshape(x.T, (-1, n))

    total_s=W@x+b
    
    loss = 0
    for n_case in range(total_s.shape[1]):
      yi = y[n_case]
      si = total_s[yi][n_case] 

      tmp_loss = 0
      for n_class in range(total_s.shape[0]):
        if y[n_case] == n_class:
          continue
        if total_s[n_class][n_case] < si:
          continue
        
        tmp_loss += (total_s[n_class][n_case] - si + 1)

      loss += tmp_loss
    
    # return SVM loss
    avg_loss = loss/total_s.shape[1]

    return avg_loss

########################################
# Part 3. kNN classification
########################################
def knn_test(X_train, y_train, X_test, y_test, n_train_sample, n_test_sample, k):
    # implement your function here
    dist = -2 * np.dot(X_train, x_test.T) + np.sum(x_test**2, axis=1) + np.sum(X_train**2, axis=1)[:, np.newaxis]

    pred_class = []
    for num_test in range(dist.shape[1]):
      # pick one test case 
      one_test = dist[:, num_test]   
      
      # pick top 3 classes from the column
      one_test_idx = np.argsort(one_test)
      top_k_idx = one_test_idx[0:3]

      top_k_class = []
      for idx in top_k_idx:    
        top_k_class.append(y_train[idx])
      
      # pick most frequent class among 3 classes
      freq_class = stats.mode(top_k_class)      
      top_freq_class = freq_class[0][0]

      # make prediction
      pred_class.append(top_freq_class)

    # count how many predictions are correct
    cnt = 0
    for pred_idx in range(len(pred_class)):
      if pred_class[pred_idx] == y_test[pred_idx]:
        cnt += 1
    
    # resulting accuracy
    rst_acc = cnt/len(pred_class)

    return rst_acc

# now lets test the model for linear models, that is, SVM and softmax
def linear_classifier_test(Wb, x_te, y_te, num_class,n_test):
    Wb = np.reshape(Wb, (-1, 1))
    dlen = len(x_te[0])
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * dlen)], (num_class, dlen))
    accuracy = 0;

    for i in range(n_test):
        # find the linear scores
        s = W @ x_te[i].reshape((-1, 1)) + b
        # find the maximum score index
        res = np.argmax(s)
        accuracy = accuracy + (res == y_te[i]).astype('uint8')

    return accuracy / n_test

# number of classes: this can be either 3 or 4
num_class = 4

# sigma controls the degree of data scattering. Larger sigma gives larger scatter
# default is 1.0. Accuracy becomes lower with larger sigma
sigma = 1.0

print('number of classes: ',num_class,' sigma for data scatter:',sigma)
if num_class == 4:
    n_train = 400
    n_test = 100
    feat_dim = 2
else:  # then 3
    n_train = 300
    n_test = 60
    feat_dim = 2

# generate train dataset
print('generating training data')
x_train, y_train = dg.generate(number=n_train, seed=None, plot=True, num_class=num_class, sigma=sigma)

# generate test dataset
print('generating test data')
x_test, y_test = dg.generate(number=n_test, seed=None, plot=False, num_class=num_class, sigma=sigma)

# set classifiers to 'svm' to test SVM classifier
# set classifiers to 'softmax' to test softmax classifier
# set classifiers to 'knn' to test kNN classifier
def classify(classifiers):
  if classifiers == 'svm':
      print('training SVM classifier...')
      w0 = np.random.normal(0, 1, (2 * num_class + num_class))
      result = minimize(svm_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))
      print('testing SVM classifier...')

      Wb = result.x
      print('accuracy of SVM loss: ', linear_classifier_test(Wb, x_test, y_test, num_class,n_test)*100,'%')

  elif classifiers == 'softmax':
      print('training softmax classifier...')
      w0 = np.random.normal(0, 1, (2 * num_class + num_class))

      result = minimize(cross_entropy_softmax_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))

      print('testing softmax classifier...')

      Wb = result.x
      print('accuracy of softmax loss: ', linear_classifier_test(Wb, x_test, y_test, num_class,n_test)*100,'%')

  else:  
      # knn
      # k value for kNN classifier. k can be either 1 or 3.
      k = 3
      print('testing kNN classifier...')
      print('accuracy of kNN loss: ', knn_test(x_train, y_train, x_test, y_test, n_train, n_test, k)*100
            , '% for k value of ', k)


classify('svm')
classify('softmax')
classify('knn')