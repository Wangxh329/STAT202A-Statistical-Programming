############################################################# 
## Stat 202A 2018 Fall - Homework 1
## Author: 
## Date : 
## Description: This script implements linear regression 
## using Gauss-Jordan elimination in both plain and
## vectorized forms
#############################################################

#############################################################
## INSTRUCTIONS: Please fill in the missing lines of code
## only where specified. Do not change function names, 
## function inputs or outputs. You can add examples at the
## end of the script (in the "Optional examples" section) to 
## double-check your work, but MAKE SURE TO COMMENT OUT ALL 
## OF YOUR EXAMPLES BEFORE SUBMITTING.
##
## Very important: Do not use the function "setwd" anywhere
## in your code. If you do, I will be unable to grade your 
## work since R will attempt to change my working directory
## to one that does not exist.
##
## Do not use the following functions for this assignment,
## except when debugging or in the optional examples section:
## 1) lm()
## 2) solve()
#############################################################


###############################################
## Function 1: Plain version of Gauss Jordan ##
###############################################


myGaussJordan <- function(A, m){
  
  # Perform Gauss Jordan elimination on A.
  # 
  # A: a square matrix.
  # m: the pivot element is A[m, m].
  # Returns a matrix with the identity matrix 
  # on the left and the inverse of A on the right. 

  #############################################
  ## FILL IN THE BODY OF THIS FUNCTION BELOW ##
  #############################################
  
  n <- dim(A)[1]
  B <- cbind(A, diag(rep(1,n))) # I_n
  
  ## Function returns the matrix B
  return(B)
  
}
rc
####################################################
## Function 2: Vectorized version of Gauss Jordan ##
####################################################

myGaussJordanVec <- function(A, m){
  
  # Perform Gauss Jordan elimination on A.
  # 
  # A: a square matrix.
  # m: the pivot element is A[m, m].
  # Returns a matrix with the identity matrix 
  # on the left and the inverse of A on the right.
  
  #############################################
  ## FILL IN THE BODY OF THIS FUNCTION BELOW ##
  #############################################
  
  ## Function returns the matrix B
  return(B)
  
}



######################################################
## Function 3: Linear regression using Gauss Jordan ##
######################################################

LinearRegression <- function(X, Y){
  
  # Find the regression coefficient estimates beta_hat
  # corresponding to the model Y = X * beta + epsilon
  # Your code must use one of the 2 Gauss Jordan 
  # functions you wrote above (either one is fine).
  # Note: we do not know what beta is. We are only 
  # given a matrix X and a vector Y and we must come 
  # up with an estimate beta_hat.
  # 
  # X: an 'n row' by 'p column' matrix of input variables.
  # Y: an n-dimensional vector of responses

  #############################################
  ## FILL IN THE BODY OF THIS FUNCTION BELOW ##
  #############################################
  
  ## Function returns the (p+1)-dimensional vector 
  ## beta_hat of regression coefficient estimates
  return(beta_hat)
  
}

########################################################
## Optional examples (comment out before submitting!) ##
########################################################

testing_Linear_Regression <- function(){
  
  ## This function is not graded; you can use it to 
  ## test out the 'myLinearRegression' function 

  # ## Define parameters
  # n    <- 3
  # p    <- 2
  
  # ## Simulate data from our assumed model.
  # ## We can assume that the true intercept is 0
  # X    <- matrix(rnorm(n * p), nrow = n)
  # print(X)
  # beta <- matrix(1:p, nrow = p)
  # Y    <- X %*% beta + rnorm(n)
  # print(Y)
  # #X   <- matrix(1:(n*p), nrow = n)
  # #beta <- matrix(1:p, nrow = p)
  # #Y    <- X %*% beta + 0.1*matrix(1:n, nrow = 5)
  
  # ## Save R's linear regression coefficients
  # R_coef  <- coef(lm(Y ~ X))
  # print(R_coef)
  # ## Save our linear regression coefficients
  # my_coef <- myLinearRegression(X, Y)
  # print(my_coef)
  
  # ## Are these two vectors different?
  # sum_square_diff <- sum((R_coef - my_coef)^2)
  # if(sum_square_diff <= 0.001){
  #   return('Both results are identical')
  # }else{
  #   return('There seems to be a problem...')
  # }
  
}

