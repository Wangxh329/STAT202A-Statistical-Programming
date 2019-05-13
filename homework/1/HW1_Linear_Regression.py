# -*- coding: utf-8 -*-
"""

 Stat 202A 2018 Fall - Homework 1-1
 Author: Xiaohan Wang
 Date : 10/08/2018

 Description: This script implements linear regression 
 using Gauss-Jordan elimination in both plain and
 vectorized forms

 INSTRUCTIONS: Please fill in the missing lines of code
 only where specified. Do not change function names, 
 function inputs or outputs. You can add examples at the
 end of the script (in the "Optional examples" section) to 
 double-check your work, but MAKE SURE TO COMMENT OUT ALL 
 OF YOUR EXAMPLES BEFORE SUBMITTING.

 Do not use any of Python's built in functions for matrix 
 inversion or for linear modeling (except for debugging or 
 in the optional examples section).
 
"""

import numpy as np

###############################################
## Function 1: Plain version of Gauss Jordan ##
###############################################

def GaussJordan(A, m):
  
    """
    Perform Gauss Jordan elimination on A.

    A: a square matrix.
    m: the pivot element is A[m, m].
    Returns a matrix with the identity matrix 
    on the left and the inverse of A on the right. 

    FILL IN THE BODY OF THIS FUNCTION BELOW 
    """

    n = A.shape[0]
    B = np.concatenate((A, np.eye(n)), axis = 1) # I_n

    for k in range(0, m):  # solve block A11(m*m)
        tmp = B[k, k]
        for j in range(0, (n*2)):
            B[k, j] = B[k, j] / tmp   
        for i in range(0, n): # traverse all rows in A
            if i != k:
                tmp = B[i, k]
                for j in range(0, (n*2)):
                    B[i, j] = B[i, j] - B[k, j] * tmp

    ## Function returns the np.array B
    return B
  
####################################################
## Function 2: Vectorized version of Gauss Jordan ##
####################################################

def GaussJordanVec(A, m):
  
    """
    Perform Gauss Jordan elimination on A using vector.

    A: a square matrix.
    m: the pivot element is A[m, m].
    Returns a matrix with the identity matrix 
    on the left and the inverse of A on the right.

    FILL IN THE BODY OF THIS FUNCTION BELOW
    """

    n = A.shape[0]
    B = np.concatenate((A, np.eye(n)), axis = 1)
    
    for k in range(0, m): # solve block A11(m*m)
        B[k, ] = B[k, ] / B[k, k]
        for i in range(0, n): # traverse all rows in A
            if i != k:
                B[i, ] = B[i, ] - B[k, ] * B[i, k]
    
    ## Function returns the np.array B
    return B
  

######################################################
## Function 3: Linear regression using Gauss Jordan ##
######################################################

def LinearRegression(X, Y):
  
    """
    Find the regression coefficient estimates beta_hat
    corresponding to the model Y = X * beta + epsilon
    Your code must use one of the 2 Gauss Jordan 
    functions you wrote above (either one is fine).
    Note: we do not know what beta is. We are only 
    given a matrix X and a vector Y and we must come 
    up with an estimate beta_hat.

    X: an 'n row' by 'p column' matrix (np.array) of input variables.
    Y: an n-dimensional vector (np.array) of responses

    FILL IN THE BODY OF THIS FUNCTION BELOW
    """

    n = X.shape[0]
    p = X.shape[1]
    X_reg = np.concatenate((np.ones((n, 1)), X), axis = 1) # add one col with all 1 before X --> dim(X_reg) = (n, p+1)
    Z = np.concatenate((X_reg, Y), axis = 1)
    A = np.dot(Z.transpose(), Z)
    B = GaussJordanVec(A, p+1)
    
    beta_hat = B[0:(p+1), p+1]
    RSS = B[p+1, p+1]
    V = B[0:(p+1), (p+2):(p+3+p)]
    sigma = RSS / (n - p - 1)
    error = V * sigma
    
    ## Function returns the (p+1)-dimensional vector (np.array) 
    ## beta_hat of regression coefficient estimates
    return beta_hat, sigma, error
  
########################################################
## Optional examples (comment out before submitting!) ##
########################################################

# def testing_Linear_Regression():
  
#     # This function is not graded; you can use it to 
#     # test out the 'myLinearRegression' function 

#     # You can set up a similar test function as was 
#     # provided to you in the R file.
  
#     ## Define parameters
#     n    = 3
#     p    = 2

#     ## Simulate data from our assumed model.
#     ## We can assume that the true intercept is 0
#     X    = np.random.random((n,p))
#     print(X)
#     beta = np.arange(1, p+1).reshape(p, 1)
#     Y    = np.dot(X, beta) + np.random.random((n,1))
#     print(Y)
#     #X   <- matrix(1:(n*p), nrow = n)
#     #beta <- matrix(1:p, nrow = p)
#     #Y    <- X %*% beta + 0.1*matrix(1:n, nrow = 5)

#     from sklearn import linear_model
#     ## Save R's linear regression coefficients
#     regr = linear_model.LinearRegression()
#     regr.fit(X, Y)
#     py_coef  = regr.coef_[0]
#     inter_coef = [regr.intercept_[0]]
#     inter_coef.append(py_coef[0])
#     inter_coef.append(py_coef[1])
#     print('py_coef')
#     print(inter_coef)
#     ## Save our linear regression coefficients
#     my_coef = LinearRegression(X, Y)[0]
#     print(my_coef)

#     ## Are these two vectors different?
#     sum_square_diff = sum(np.square(inter_coef - my_coef))
#     if sum_square_diff <= 0.001:
#         print('Both results are identical')
#     else:
#         print('There seems to be a problem...')

# testing_Linear_Regression()
