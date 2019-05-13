#########################################################
## Stat 202A - Homework 4
## Author: Xiaohan Wang
## Date: 10/24/2018
## Description: This script implements logistic regression
## using iterated reweighted least squares using the code 
## we have written for linear regression based on QR 
## decomposition
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

myLM <- function(X, Y){
  
  ## Perform the linear regression of Y on X
  ## Input: 
  ## X is an n x p matrix of explanatory variables
  ## Y is an n dimensional vector of responses
  ## Use myQR inside of this function
  
  ########################
  ## FILL IN CODE BELOW ##
  ########################  
    
  n <- dim(X)[1]
  p <- dim(X)[2]
  Z <- cbind(X, Y)
  if (n < p + 1) { # input cannot do QR decomposition
    return(-1)
  }
  
  result <- myQR(Z)
  R <- result$R
  R1 <- R[1:p, 1:p]
  Y1 <- R[1:p, p+1]
  Y2 <- R[(p+1):n, p+1]
  beta_ls <- solve(R1, Y1)
  
  ## Function returns the 1 x p vector beta_ls, notice this version do not add intercept.
  ## the least squares solution vector
  return(beta_ls)
  
}

######################################
## Function 3: Logistic regression  ##
######################################

## Expit/sigmoid function
expit <- function(x){
  1 / (1 + exp(-x))
}

myLogisticSolution <- function(X, Y){

  ########################
  ## FILL IN CODE BELOW ##
  ########################
    
  n <- nrow(X)
  p <- ncol(X)    
  beta <- matrix(rep(0, p), nrow = p)
  epsilon <- 1e-6
    
  repeat {
    eta <- X %*% beta
    pr <- expit(eta)
    w <- pr * (1 - pr)
    Z <- eta + (Y - pr) / w
    if (NaN %in% Z) {  # eta is big -> pr = 0 -> w = 0
        break
    }
    sw <- sqrt(w)
    mw <- matrix(sw, n, p)
    X_work <- mw * X
    Y_work <- sw * Z
    beta_new <- myLM(X_work, Y_work)
    err <- sum(abs(beta_new - beta))
    
    beta <- beta_new
    if (err < epsilon) {
      break  
    }
  }

  return(beta)
    
}

##################################################
## Function 4: Eigen decomposition based on QR  ##
##################################################
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

###################################################
## Function 5: PCA based on Eigen decomposition  ##
###################################################
myPCA <- function(X) {
  
  ## Perform PCA on matrix A using your eigen decomposition.
  ## Input:
  ## X: Input Matrix with dimension n * p

  A <- t(X) %*% X
  result <- myEigen_QR(A)
  D <- result$D
  Q <- result$V
  Z <- X %*% Q

  ## Output : 
  ## Q : basis matrix, p * p which is the basis system.
  ## Z : data matrix with dimension n * p based on the basis Q.
  ## It should match X = Z %*% Q.T. Please follow lecture notes.

  return(list("Q" = Q, "Z" = Z, "L" = diag(D)))
}

# # Simulation
# n <- 100
# p <- 4

# X    <- matrix(rnorm(n * p), nrow = n)
# beta <- c(12, -2,-3, 4)
# eta <- X %*% beta
# Y    <- 1 * (runif(n) < expit(eta))

# ## test myPCA
# print('==== test PCA ====')
# result <- myPCA(X)
# Z <- result$Z
# L <- result$L
# ZTZ <- t(Z) %*% Z
# print(ZTZ)
# print(L)

# ## test logistic regression
# ## Our solution
# print('==== test logistic regression ====')
# logistic_beta <- myLogisticSolution(X, Y)
# print(logistic_beta)    

# ## R's solution
# print(coef(glm(Y ~ X + 0, family = binomial(link = 'logit'))))