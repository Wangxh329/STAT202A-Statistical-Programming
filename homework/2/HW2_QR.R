#########################################################
## Stat 202A - Homework 2
## Author: Xiaohan Wang
## Date : 10/16/2018
## Description: This script implements QR decomposition,
## linear regression, and eigen decomposition / PCA 
## based on QR.
#########################################################

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

##################################
## Function 1: QR decomposition ##
##################################

myQR <- function(A){
  
  ## Perform QR decomposition on the matrix A
  ## Input: 
  ## A, an n x m matrix
  
  ########################
  ## FILL IN CODE BELOW ##
  ########################  
  
  n <- dim(A)[1]
  m <- dim(A)[2]
  R <- A
  Q <- diag(rep(1, n))

  for (k in 1:(m-1)) {
    x <- rep(0, n)
    x[k:n] <- R[k:n, k]
    v <- x
    v[k] <- x[k] + sign(x[k]) * norm(x, type="2")
    s <- norm(v, type="2")
    
    if (s != 0) {
      u <- v / s
      R <- R - 2 * (u %*% (t(u) %*% R))
      Q <- Q - 2 * (u %*% (t(u) %*% Q))
    }
  }  

  ## Function should output a list with Q.transpose and R
  ## Q is an orthogonal n x n matrix
  ## R is an upper triangular n x m matrix
  ## Q and R satisfy the equation: A = Q %*% R
  return(list("Q" = t(Q), "R" = R))
  
}

###############################################
## Function 2: Linear regression based on QR ##
###############################################

myLinearRegression <- function(X, Y){
  
  ## Perform the linear regression of Y on X
  ## Input: 
  ## X is an n x p matrix of explanatory variables
  ## Y is an n dimensional vector of responses
  ## Do NOT simulate data in this function. n and p
  ## should be determined by X.
  ## Use myQR inside of this function
  
  ########################
  ## FILL IN CODE BELOW ##
  ########################  
  
  n <- dim(X)[1]
  p <- dim(X)[2]
  Z <- cbind(rep(1, n), X, Y)
  if (n < p + 2) { # input cannot do QR decomposition
    return(-1)
  }
  
  result <- myQR(Z)
  R <- result$R
  R1 <- R[1:(p+1), 1:(p+1)]
  Y1 <- R[1:(p+1), p+2]
  Y2 <- R[(p+2):n, p+2]
  beta_hat <- solve(R1, Y1)
  RSS <- norm(Y2, type="2")^2
  sigma <- RSS / (n - p - 1)
  variance <- sigma * solve(t(R1) %*% R1)
  error <- sqrt(diag(variance))

  ## Function returns the 1 x (p + 1) vector beta_ls, 
  ## the least squares solution vector
  return(list(beta_hat=beta_hat, sigma=sigma, error=error))
  
}

##################################
## Function 3: PCA based on QR  ##
##################################
myEigen_QR <- function(A, numIter = 1000) {
  
  ## Perform PCA on matrix A using your QR function, myQRC.
  ## Input:
  ## A: Square matrix
  ## numIter: Number of iterations
  
  ########################
  ## FILL IN CODE BELOW ##
  ######################## 

  r <- dim(A)[1]
  V <- matrix(rnorm(r*r), nrow=r)

  for (i in 1:numIter) {
    result <- myQR(V)
    V <- A %*% result$Q
  }
  result <- myQR(V)
  Q <- result$Q
  R <- result$R
  
  ## Function should output a list with D and V
  ## D is a vector of eigenvalues of A
  ## V is the matrix of eigenvectors of A (in the 
  ## same order as the eigenvalues in D.)
  
  return(list("D" = diag(R), "V" = Q))
}

tlr <- function(){
  
#   ## This function is not graded; you can use it to 
#   ## test out the 'myLinearRegression' function 

#   ## Define parameters
#   n    <- 10
#   p    <- 5
  
#   ## Simulate data from our assumed model.
#   ## We can assume that the true intercept is 0
#   X    <- matrix(rnorm(n * p), nrow = n)
#   beta <- matrix(1:p, nrow = p)
#   Y    <- X %*% beta + rnorm(n)
  
#   ## 1. Test coefficient 
#   ## Save R's linear regression coefficients
#   R_coef  <- coef(lm(Y ~ X))
#   print(R_coef)
  
#   ## Save our linear regression coefficients
#   my_coef <- myLinearRegression(X, Y)[['beta_hat']]
#   print(my_coef)
  
#   ## Are these two vectors different?
#   sum_square_diff <- sum((R_coef - my_coef)^2)
#   if(sum_square_diff <= 0.001){
#     return('Both results are identical - coefficient')
#   }else{
#     return('There seems to be a problem... - coefficient')
#   }
  
#   ## 2. Test standard error
#   ## Save R's linear regression standard error
#   R_error <- coef(summary(lm(Y ~ X)))[,"Std. Error"]
#   print(R_error)
   
#   ## Save our linear regression standard error
#   my_error <- myLinearRegression(X, Y)[['error']]
#   print(my_error)
    
#   ## Are these two vectors different?
#   sum_square_diff <- sum((R_error - my_error)^2)
#   if(sum_square_diff <= 0.001){
#     return('Both results are identical - error')
#   }else{
#     return('There seems to be a problem... - error')
#   }
    
#   ## 3. Test eigen_qr
#   ## Save R's eigen
#   R_eigen <- eigen(X)
#   R_values <- R_eigen$values
#   print(R_values)
#   R_vectors <- R_eigen$vectors
#   print(R_vectors)
#   ## Save our results
#   my_eigen <- myEigen_QR(X)
#   my_values <- my_eigen$D
#   print(my_values)
#   my_vectors <- my_eigen$V
#   print(my_vectors)
  
}

# tlr()
