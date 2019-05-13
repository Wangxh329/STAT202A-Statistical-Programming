#############################################################
## Stat 202A - Homework 4
## Author: Xiaohan Wang
## Date : 11/4/2018
## Description: This is HW4. You are require to implement a 
## 2-layer neural network in 3 way to classify mnist.
#############################################################

#############################################################
# INSTRUCTIONS: Please fill in the missing lines of code
# only where specified. You only need to fill "# Define it"
# and "FILL IN CODE HERE" area.
#
# You are required to implement neural network in three way:
# 1) Write from scratch, following professor's code. Change output layer to 10 classes
# 2) Write it through tensorflow
# 3) Write it through pyTorch 
# For each function, you are given a dataset and required to output parameters: W1, b1, W2, b2 stand two layer weights and its bias.
# Try to run main_test() to evaluate your code. evaluate() return the accuracy of your parameter. Receive full score if the accuracy is (>80% for scratch version, >95% for tensorflow and pytorch version), receive half score if the accuracy is >20%. Notice on grading stage, the input dataset may changes. 
# Function (1) worth 60% score, (2)(3) worth 30% each. That mean you can select only one function to finish to have 90% score. 

## The structure of network should be: 
## Input data : n * p ( p = 784 )
## Input >> FC >> relu >> FC >> Output
## Output data : n * 10 >> 
## See evaluate function for detail structure.
## 2) solve()
#############################################################

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import torch
import numpy as np

###############################################
## Function 1: Write 2-layer NN from scratch ##
###############################################

def my_NN_scratch(mnist):

    X_test = mnist.test.images
    Y_test = mnist.test.labels
    ntest = X_test.shape[0]
    num_hidden = 100
    num_iterations = 3000  
    learning_rate = 1e-3

    #######################
    ## FILL IN CODE HERE ##
    #######################

    alpha = np.random.random(size = [X_test.shape[1] + 1, num_hidden]) * 0.0001
    beta = np.random.random(size = [num_hidden + 1, 10]) * 0.0001

    for it in range(num_iterations):

        batch_xs, batch_ys = mnist.train.next_batch(100)
        #######################
        ## FILL IN CODE HERE ##
        #######################
        batch_xs_concat = np.concatenate((np.repeat(1, 100, axis=0).reshape((100, 1)),batch_xs), axis=1)
        y1 = batch_xs_concat.dot(alpha)
        y1 = np.maximum(y1, 0)

        y1_concat = np.concatenate((np.repeat(1, 100, axis=0).reshape(100, 1), y1), axis = 1)
        y2 = y1_concat.dot(beta)
        e = batch_ys - y2
        de_dy1 = -2 * np.dot(e, np.transpose(beta))

        de_dbeta = -2 * np.transpose(y1_concat).dot(e)
        beta -= learning_rate * de_dbeta
        
        for col in range(num_hidden):
            sigma = np.ones(num_hidden).reshape((num_hidden, 1))
            sigma[np.where(y1[:, col] == 0)] = 0
            de_dtemp = np.multiply(sigma, de_dy1[:, col + 1].reshape((num_hidden, 1)))
            dtemp_dcol = batch_xs_concat.T
            de_dcol = np.dot(dtemp_dcol, de_dtemp)
            alpha[:, col] -= (de_dcol * learning_rate).flatten()

        if it % 100 == 0:  
            print("Iteration ", it)
    return alpha[1:, :], alpha[0, :], beta[1:, :], beta[0, :]

#########################################################################
## Function 1.1: Write 2-layer NN from scratch with 2-class classifier ##
#########################################################################

def accuracy(p, y):
    """
    Calculate the accuracy of predictions against truth labels.

        Accuracy = # of correct predictions / # of data.

    Args:
        p: predictions of shape (n, 1)
        y: true labels of shape (n, 1)

    Returns:
        accuracy: The ratio of correct predictions to the size of data.
    """
    return np.mean((p > 0.5) == (y == 1))

def my_NN_2class(mnist_m):

    X_train = mnist_m.train.images
    Y_train = mnist_m.train.labels
    idx = np.where(Y_train < 2)
    X_train = X_train[idx[:1000]]
    n = X_train.shape[0]
    Y_train = Y_train[idx[:1000]].reshape((n, 1))
    X_test = mnist_m.test.images
    Y_test = mnist_m.test.labels
    idx = np.where(Y_test < 2)
    X_test = X_test[idx]
    ntest = X_test.shape[0]
    Y_test = Y_test[idx].reshape((ntest, 1))

    #######################
    ## FILL IN CODE HERE ##
    #######################
    # set parameters
    num_hidden = 100
    num_iterations = 1000
    learning_rate = 1e-1
    # concatenate 1 column of 1s
    p = X_train.shape[1] + 1 # 1001
    X_train1 = np.concatenate((np.repeat(1, n, axis=0).reshape((n, 1)), X_train), axis=1)
    X_test1 = np.concatenate((np.repeat(1, ntest, axis=0).reshape((ntest, 1)), X_test), axis=1)
    # initial parameters
    alpha = np.random.normal(scale=0.3, size=(p, num_hidden))
    beta = np.random.normal(scale=0.3, size=(num_hidden + 1, 1))
    acc_train = np.repeat(0., num_iterations)
    acc_test = np.repeat(0., num_iterations)

    for it in range(num_iterations):
        
        ###### training ######
        # update beta
        Z = np.maximum(0, X_train1.dot(alpha)) # ReLU
        Z1 = np.concatenate((np.repeat(1, n, axis=0).reshape((n, 1)), Z), axis=1)
        pr = 1 / (1 + np.exp((-1) * Z1.dot(beta)))

        dbeta1 = np.repeat(1, n, axis=0).reshape((1, n))
        dbeta2 = (np.repeat((Y_train - pr), (num_hidden + 1)).reshape((n, (num_hidden + 1))) * Z1) / n
        dbeta = dbeta1.dot(dbeta2)
        beta += learning_rate * np.transpose(dbeta)

        # update alpha
        for k in range(num_hidden):
            da = (Y_train - pr) * beta[k + 1] * Z[:, k].reshape((n, 1)) * (1 - Z[:, k].reshape((n, 1)))
            dalpha1 = np.repeat(1, n, axis=0).reshape((1, n))
            dalpha2 = (np.repeat(da, p).reshape((n, p)) * X_train1) / n
            dalpha = dalpha1.dot(dalpha2)
            alpha[:, k] = (alpha[:, k].reshape((p, 1)) + learning_rate * np.transpose(dalpha)).reshape((1, p))
        
        # calculate accuracy of training
        acc_train[it] = accuracy(pr, Y_train)
            
        ###### testing ######
        Ztest = np.maximum(0, X_test1.dot(alpha))
        Ztest1 = np.concatenate((np.repeat(1, ntest, axis=0).reshape((ntest, 1)), Ztest), axis=1)
        prtest = 1 / (1 + np.exp((-1) * Ztest1.dot(beta)))
        acc_test[it] = accuracy(prtest, Y_test)

        if it % 50 == 0:
            print("Iteration ", it, "Training Accuracy: ", acc_train[it], "Testing Accuracy: ", acc_test[it])
        
    #######################
    ## FILL IN CODE HERE ##
    #######################

    return alpha, beta, acc_train, acc_test

################################################
## Function 2: Write 2-layer NN by Tensorflow ##
################################################

def my_NN_tensorflow(mnist):

    num_hidden = 100
    x = tf.placeholder(tf.float32, [None, 784])
    num_iterations = 4000
    W1 = tf.Variable(tf.random_uniform([784,num_hidden]) * 0.01)
    b1 = tf.Variable(tf.zeros([1, num_hidden]))
    W2 = tf.Variable(tf.random_uniform([num_hidden, 10]) * 0.01)
    b2 = tf.Variable(tf.zeros([1, 10]))
    z = tf.nn.relu(tf.matmul(x, W1) + b1)
    y = tf.matmul(z, W2) + b2

    y_ =  tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y)) 
    train_step = tf.train.GradientDescentOptimizer(0.002).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    for epoch in range(num_iterations):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        res = sess.run([train_step, W1, b1, W2, b2], feed_dict={
            x: batch_xs, y_:batch_ys})  # Define it
        if epoch % 100 == 0:  
            print("Iteration ", epoch)
    print(sess.run(accuracy, feed_dict={
            x: mnist.test.images, y_: mnist.test.labels}))
    W1_e, b1_e, W2_e, b2_e = W1.eval(), b1.eval(), W2.eval(), b2.eval()
    sess.close()

    return W1_e, b1_e, W2_e, b2_e

###############################################
## Function 3: Write 2-layer NN by pyTorch   ##
###############################################

def my_NN_pytorch(mnist_m):

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Sequential(                    
                torch.nn.Linear(784, 100), 
                torch.nn.ReLU(),
            )
            self.fc2 = torch.nn.Sequential(
                torch.nn.Linear(100, 10),
            )

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    net = Net()
    net.zero_grad()
    Loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(5000):  # loop over the dataset multiple times

        batch_xs, batch_ys = mnist_m.train.next_batch(100)
        batch_ys_vector = np.zeros((100, 10))
        for i in range(100):
            batch_ys_vector[i][batch_ys[i]] = 1
        #######################
        ## FILL IN CODE HERE ##
        #######################
        tensor_xs = torch.from_numpy(batch_xs)
        tensor_ys = torch.FloatTensor(batch_ys_vector)
        y_predict = net(tensor_xs)
        loss = Loss(y_predict, tensor_ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Training Epoch: {}'.format(epoch))

    params = list(net.parameters())
    return params[0].detach().numpy().T, params[1].detach().numpy(), \
        params[2].detach().numpy().T, params[3].detach().numpy()


def evaluate(W1, b1, W2, b2, data):

    inputs = data.test.images
    outputs = np.dot(np.maximum(np.dot(inputs, W1) + b1, 0), W2) + b2
    predicted = np.argmax(outputs, axis=1)
    accuracy = np.sum(predicted == data.test.labels)*100 / outputs.shape[0]
    print('Accuracy of the network on test images: %.f %%' % accuracy)
    return accuracy


def main_test():

    mnist = input_data.read_data_sets('input_data', one_hot=True)
    mnist_m = input_data.read_data_sets('input_data', one_hot=False)
    W1, b1, W2, b2 = my_NN_scratch(mnist)
    evaluate(W1, b1, W2, b2, mnist_m)
    W1, b1, W2, b2 = my_NN_tensorflow(mnist)
    evaluate(W1, b1, W2, b2, mnist_m)
    W1, b1, W2, b2 = my_NN_pytorch(mnist_m)
    evaluate(W1, b1, W2, b2, mnist_m)
