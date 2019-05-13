#############################################################
## Stat 202A - Homework 7
## Author: Xiaohan Wang
## Date: 12/07/2018
## Description: This script implements the lasso
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
#############################################################

## Source your Rcpp file (put in the name of your Rcpp file)
library(Rcpp)
sourceCpp("./HW7.cpp")

##################################
## Function 1: Ridge regression ##
##################################

myRidge <- function(X, Y, lambda, use_QR = FALSE, use_C = TRUE){
  
  # Perform ridge regression of Y on X.
  # 
  # X: an n x p matrix of explanatory variables.
  # Y: an n vector of dependent variables. Y can also be a 
  # matrix, as long as the function works.
  # lambda: regularization parameter (lambda >= 0)
  # Returns beta, the ridge regression solution.
  
  ##################################
  ## FILL IN THIS SECTION OF CODE ##
  ##################################

  n <- nrow(X)
  p <- ncol(X)

  Z <- cbind(rep(1, n), X, Y)
  A <- t(Z) %*% Z

  D <- diag(rep(lambda, p+2)) # add penalty term
  D[1, 1] <- 0
  D[p+2, p+2] <- 0
  A <- A + D

  S <- mySweepC(A, p+1)
  beta_ridge <- S[1:(p+1), p+2]
  
  ## Function should output the vector beta_ridge, the 
  ## solution to the ridge regression problem. beta_ridge
  ## should have p + 1 elements.
  return(beta_ridge)
  
}

####################################################
## Function 2: Piecewise linear spline regression ##
####################################################

mySpline <- function(x, Y, lambda = 1, p = 100, use_QR = FALSE, use_C = TRUE){
  
  # Perform spline regression of Y on X.
  # 
  # x: An n x 1 vector or matrix of explanatory variables.
  # Y: An n x 1 vector of dependent variables. Y can also be a 
  # matrix, as long as the function works.
  # lambda: regularization parameter (lambda >= 0)
  # p: Number of cuts to make to the x-axis.
  
  ##################################
  ## FILL IN THIS SECTION OF CODE ##
  ##################################

  n <- length(x)
  X <- matrix(x, nrow = n)

  for (k in (1:(p-1))/p) {
    X <- cbind(X, (x>k)*(x-k))
  }

  beta_spline = myRidge(X, Y, lambda)
  Yhat = cbind(rep(1, n), X) %*% beta_spline
  
  ## Function should a list containing two elements:
  ## The first element of the list is the spline regression
  ## beta vector, which should be p + 1 dimensional (here, 
  ## p is the number of cuts we made to the x-axis).
  ## The second element is y.hat, the predicted Y values
  ## using the spline regression beta vector. This 
  ## can be a numeric vector or matrix.
  output <- list(beta_spline = beta_spline, predicted_y = Yhat)
  return(output)
  
}

#####################################
##   Function 3: Visualizations    ##
#####################################

myPlotting <- function(){

  # Write code to plot the result. 

  #######################
  ## FILL IN CODE HERE ##
  #######################

#   # ============================================================ #
#   # (1) Explore the effect of regularization on the smoothness   #
#   #     of the fitted curve (Output curve in one plot)           #
#   # ============================================================ #

#   # generate random data
#   n = 50
#   p = 200
#   sigma = 0.1
#   x = runif(n)
#   x = sort(x)
#   Y = x^2 + rnorm(n)*sigma
    
#   # plot original data
#   plot(x, Y, ylim = c(-.2, 1.2), col = "red", xlab = "x", ylab ="Y/Y_hat")
    
#   # change lambda
#   colors = c('deepskyblue', 'coral', 'limegreen', 'gold', 'cyan3', 'deeppink', 'mediumorchid', 'lightpink')
#   all_lambda = rep(1, 8)
#   lambda = 0.0001
#   errors = rep(1, 8)
#   for (i in 1:8) {
#     all_lambda[i] = lambda
#     output = mySpline(x, Y, lambda, p)
#     lines(x, output[['predicted_y']], ylim = c(-.2, 1.2), col = colors[i], type = 'l', lwd=2)
#     errors[i] = sum((Y-output[['predicted_y']])^2)
#     lambda = lambda * 10
#   }
#   legends = c("lambda = 0.0001", "lambda = 0.001", "lambda = 0.01", "lambda = 0.1", "lambda = 1", "lambda = 10", "lambda = 100", "lambda = 1000")
#   legend("topleft", pch=c(15,15), legend=legends, col=colors, text.col=colors, bty="n")
  
#   # ============================================================ #
#   # (2) Estimator error in terms of the L2 difference between    #
#   #     the true curve and the learned curve.                    #
#   # ============================================================ #
#   plot(c(1:8), errors, xlab="lambda", ylab="error", xaxt="n", type="o", lwd=2, col="deeppink")
#   axis(1,at=c(1:8), labels=all_lambda)
  
#   # ============================================================ #
#   # (3) Plot solution path of Lasso                              #
#   # ============================================================ #
  
#   # generate random data
#   lambda_all = matrix(c(100:1)*10, nrow=100)
#   X = matrix(rnorm(n*p), nrow=n)
#   beta_true = matrix(rep(0, p), nrow = p)
#   beta_true[1:10] = 1:10
#   Y = X %*% beta_true + rnorm(n)
    
#   # plot solution path
#   beta_all = myLassoC(X, Y, lambda_all)
#   matplot(t(matrix(rep(1, p), nrow = 1)%*%abs(beta_all)), t(beta_all), type = 'l', main="Solution Path of Lasso", xlab="L1 of beta_i, i[1,100]", ylab="beta")

#   # ============================================================ #
#   # (4) Plot solution path of Epsilon-boosting                   #
#   # ============================================================ #
#   beta_all = myBoostingC(X, Y, lambda_all)
#   matplot(t(matrix(rep(1, p), nrow = 1)%*%abs(beta_all)), t(beta_all), type = 'l', main="Solution Path of Epsilon-boosting", xlab="L1 of beta_i, i[1,5000]", ylab="beta")

}

# myPlotting()

