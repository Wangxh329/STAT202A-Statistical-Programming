/*
####################################################
## Stat 202A - Homework 7
## Author: Xiaohan Wang
## Date: 12/07/2018 
## Description: This script implements Sweep and lasso
####################################################
 
###########################################################
## INSTRUCTIONS: Please fill in the missing lines of code
## only where specified. Do not change function names, 
## function inputs or outputs. MAKE SURE TO COMMENT OUT ALL 
## OF YOUR EXAMPLES BEFORE SUBMITTING.
##
## Very important: Do not change your working directory
## anywhere inside of your code. If you do, I will be unable 
## to grade your work since R will attempt to change my 
## working directory to one that does not exist.
###########################################################
 
*/ 

# include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
  Sample function: QR decomposition 
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

// [[Rcpp::export()]]
List myQRC(const mat A)
{

  /*
  Perform QR decomposition on the matrix A
  Input: 
  A, an n x m matrix (mat)
  */

  int n = A.n_rows;
  int m = A.n_cols;
  mat R = A;
  mat Q = mat(n, n, fill::zeros);
  for (int i = 0; i < n; i++)
    Q(i, i) = 1;
  int mm = (n > m) ? m : n;
  for (int k = 0; k < mm; k++)
  {
    mat X = mat(n, 1, fill::zeros);
    for (int i = k; i < n; i++)
    {
      X(i) = R(i, k);
    }
    mat V = X;
    double norm = 0;
    for (int i = 0; i < n; i++)
      norm = norm + X(i) * X(i);
    V(k) = X(k) + sign(X(k)) * sqrt(norm);
    norm = 0;
    for (int i = 0; i < n; i++)
      norm = norm + V(i) * V(i);
    mat U = V / sqrt(norm);
    R = R - 2 * ((U * U.t()) * R);
    Q = Q - 2 * ((U * U.t()) * Q);
  }

  List output;
  // Function should output a List 'output', with
  // Q.transpose and R
  // Q is an orthogonal n x n matrix
  // R is an upper triangular n x m matrix
  // Q and R satisfy the equation: A = Q %*% R
  output["Q"] = Q.t();
  output["R"] = R;
  return (output);
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~ 
   Problem 1: Sweep operator 
   ~~~~~~~~~~~~~~~~~~~~~~~~~ */

// [[Rcpp::export()]]
mat mySweepC(const mat A, int m){
  
  /*
  Perform a SWEEP operation on A with the pivot element A[m,m].
  
  A: a square matrix (mat).
  m: the pivot element is A[m, m]. 
  Returns a swept matrix B (which is m by m).
  
  Note the "const" in front of mat A; this is so you
  don't accidentally change A inside your code.
  
  #############################################
  ## FILL IN THE BODY OF THIS FUNCTION BELOW ##
  #############################################
  
  */

  mat B = A;
  int n = B.n_rows;
  
  for (int k = 0; k < m; k++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if ((i != k) & (j != k)) {
          B(i, j) = B(i, j) - B(i, k) * B(k, j) / B(k, k);
        }
      }
    }
    
    for (int i = 0; i < n; i++) {
      if (i != k) {
        B(i, k) = B(i, k) / B(k, k);
      }
    }
    
    for (int j = 0; j < n; j++) {
      if (j != k) {
        B(k, j) = B(k, j) / B(k, k);
      }
    }
    
    B(k, k) = - 1 / B(k, k);
  }
  
  // Return swept matrix B
  return(B);
  
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~ 
   Problem 2: Path for Lasso 
   ~~~~~~~~~~~~~~~~~~~~~~~~~ */

// [[Rcpp::export()]]
mat myLassoC(const mat X, mat Y, mat lambda_all)
{

  /*
  Find the lasso solution path for various values of
  the regularization parameter lambda.
  
  X: n x p matrix of explanatory variables.
  Y: n dimensional response vector
  lambda: Vector of regularization parameters. Make sure
  to sort lambda in decreasing order for efficiency.
  
  Returns a matrix containing the lasso solution vector
  beta for each regularization parameter.
  
  #############################################
  ## FILL IN THE BODY OF THIS FUNCTION BELOW ##
  #############################################
  
  */

  // Sort lambda_all
  std::sort(lambda_all);
  std::reverse(lambda_all);

  // Parameters
  int n = X.n_rows;
  int p = X.n_cols;
  int L = lambda_all.n_rows;

  // Constants
  int T = 10;

  // beta
  mat beta = mat(p, 1, fill::zeros);
  mat beta_all = mat(p, L, fill::zeros);

  mat R = Y;
  mat ss = mat(p, 1, fill::zeros);
  for (int j = 0; j < p; j++) {
    for (int i = 0; i < n; i++) {
      ss(j) += X(i, j) * X(i, j);
    }
  }

  for (int l = 0; l < L; l++) {
    int lambda = lambda_all(l);
    for (int t = 0; t < T; t++) {
      for (int j = 0; j < p; j++) {
        double db = 0;
        for (int i = 0; i < n; i++) {
          db += R(i) * X(i, j);
        }
        db /= ss(j);

        double b = beta(j) + db;
        b = sign(b) * max(0, abs(b) - lambda / ss(j));
        db = b - beta(j);

        for (int i = 0; i < n; i++) {
          R(i) -= X(i, j) * db;
        }

        beta(j) = b;
        beta_all(j, l) = beta(j);
      }
    }
  }

  // Return a matrix beta_all with (p+1) x length(lambda_all)
  return (beta_all);
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~ 
   Problem 3: Path for e-Boosting 
   ~~~~~~~~~~~~~~~~~~~~~~~~~ */

// [[Rcpp::export()]]
mat myBoostingC(const mat X, mat Y, mat lambda_all)
{

  /*
  Find the epsilon boosting solution path for various values of
  the regularization parameter lambda.
  
  X: n x p matrix of explanatory variables.
  Y: n dimensional response vector
  lambda: Vector of regularization parameters. Make sure
  to sort lambda in decreasing order for efficiency.
  
  Returns a matrix containing the Boosting solution vector
  beta for each regularization parameter.
  
  #############################################
  ## FILL IN THE BODY OF THIS FUNCTION BELOW ##
  #############################################
  
  */

  // Parameters
  int n = X.n_rows;
  int p = X.n_cols;

  // constant
  int T = 5000;
  double epsilon = 0.0001;

  // beta
  mat beta = mat(p, 1, fill::zeros);
  mat beta_all = mat(p, T, fill::zeros);
  mat db = mat(p, 1, fill::zeros);

  mat R = Y;
  for (int t = 0; t < T; t++) {
    for (int j = 0; j < p; j++) {
      for (int i = 0; i < n; i++) {
        db(j) += R(i) * X(i, j);
      }
    }

    // find max col
    int col = 0;
    double max = db(0);
    for (int j = 0; j < p; j++) {
      if (abs(db(j)) > max) {
        max = abs(db(j));
        col = j;
      }
    }

    beta(col) += db(col) * epsilon;

    for (int i = 0; i < n; i++) {
      R(i) -= X(i, col) * db(col) * epsilon;
    }

    for (int j = 0; j < p; j++) {
      beta_all(j, t) = beta(j);
    }
  }

  // Return a matrix beta_all with (p+1) x length(lambda_all)
  return (beta_all);
}
