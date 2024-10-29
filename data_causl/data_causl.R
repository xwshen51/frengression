## Import packages
library(glmnet)
library(zeallot) #enable %<-%
library(mvtnorm)
library(causl)

logit <- function(p){
  return(log(p/(1-p)))
}


# an example of simulating data from causl
# I, X , O, S stand for instrumental varable, confounder, outcome variable, spurious variable.
# expectation of potential outcomes and propensity scores are set to be linear
# strength_instr, strength_conf stand for coefficients in propensity score model for I and X, respectively.
# strength outcome stand for coefficients in potential outcome model for X and O.
# Z = (I,X,O,S); Z \overset{i.i.d}{\sim} N(0,1) 
# A \sim bernoulli(pi(Z)), pi(Z) = logit( \sum_i^{nI} strength_instr_i * I_i + \sum_i^{nX} strength_conf_i * X_i )
# Y|do(A) \sim N(ate*A,1)
# copula: gaussian copula, with coefficient to be strength_outcome
# beta_cov: constant shift. set = 0 for simplification.
data.causl <- function(n=10000, nI = 3, nX= 1, nO = 1, nS = 1, ate = 2, beta_cov = 0, strength_instr = 3, strength_conf = 1, strength_outcome = 0.2){

  forms <- list(list(), A ~ 1, Y ~ A, ~ 1)
  # family: 5: bernoulli for treatment. 1: gaussian for outcome and covariates.
  fam <- list(rep(1,nI+nX+nO+nS), 5, 1, 1)
  pars = list()

  # specify the formula for each covariates
  ## for I
  for (i in seq_len(nI)) {
    forms[[1]] <- c(forms[[1]], as.formula(paste0("I", i, " ~ 1")))
    pars[paste0("I", i)]= list(list(beta = beta_cov, phi = 1))
  }
  ## and for X,O,S
  for (i in seq_len(nX)) {
    forms[[1]] <- c(forms[[1]], as.formula(paste0("X", i, " ~ 1")))
    pars[paste0("X", i)]= list(list(beta = beta_cov, phi = 1))
  }


  for (i in seq_len(nO)) {
    forms[[1]] <- c(forms[[1]], as.formula(paste0("O", i, " ~ 1")))
    pars[paste0("O", i)]= list(list(beta = beta_cov, phi = 1))
  }

  for (i in seq_len(nS)) {
    forms[[1]] <- c(forms[[1]], as.formula(paste0("S", i, " ~ 1")))
    pars[paste0("S", i)]= list(list(beta = beta_cov, phi = 1))
  }

  # specify the formula for A given covariates
  ## for I
  for (i in seq_len(nI)) {
    forms[[2]] <- update.formula(forms[[2]], paste0("A", " ~ . + I",i))
  }
  ## and for X
  for (i in seq_len(nX)) {
    forms[[2]] <- update.formula(forms[[2]], paste0("A", " ~ . + X",i))
  }


  # parameter for copula
  parY <- c(rep(list(list(beta=strength_outcome)), nX+nO), rep(list(list(beta=0)), nI+nS))
  names(parY) <- c(paste0("X", seq_len(nX)), paste0("O", seq_len(nO)), paste0("I", seq_len(nI)), paste0("S", seq_len(nS)))
  pars$cop = list(Y=parY)

  # each variable should be specified in pars
  pars$A$beta <- c(0, rep(strength_instr,nI), rep(strength_conf,nX))
  pars$Y$beta <- c(0, ate)
  pars$Y$phi <- 1

  df = rfrugalParam(n=n, formulas=forms, pars=pars, family=fam)
  df['propen'] = plogis( rowSums(c(rep(strength_instr,nI), rep(strength_conf,nX)) * df[,c(1:(nI+nX))]))
  p = nX + nI + nO + nS
  colnames(df) = c(paste("X", c(1 : p), sep=""), 'A', 'y', 'propen')
return(df)
} 