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
data.causl <- function(n=10000, nI=3, nX=1, nO=1, nS=1, ate=2, beta_cov=0, strength_instr=3, strength_conf=1, strength_outcome=0.2, binary_intervention=TRUE){
  
  forms <- list(list(), A ~ 1, Y ~ A, ~ 1)
  
  if(binary_intervention){
    fam <- list(rep(1, nI + nX + nO + nS), 5, 1, 1)
  } else {
    fam <- list(rep(1, nI + nX + nO + nS), 1, 1, 1)
  }
  
  pars <- list()
  
  # Specify the formula and parameters for each covariate type
  ## Instrumental variables (I)
  if (nI > 0) {
    for (i in seq_len(nI)) {
      forms[[1]] <- c(forms[[1]], as.formula(paste0("I", i, " ~ 1")))
      pars[paste0("I", i)] <- list(list(beta = beta_cov, phi = 1))
    }
  }
  
  ## Confounders (X)
  if (nX > 0) {
    for (i in seq_len(nX)) {
      forms[[1]] <- c(forms[[1]], as.formula(paste0("X", i, " ~ 1")))
      pars[paste0("X", i)] <- list(list(beta = beta_cov, phi = 1))
    }
  }
  
  ## Outcome variables (O)
  if (nO > 0) {
    for (i in seq_len(nO)) {
      forms[[1]] <- c(forms[[1]], as.formula(paste0("O", i, " ~ 1")))
      pars[paste0("O", i)] <- list(list(beta = beta_cov, phi = 1))
    }
  }
  
  ## Spurious variables (S)
  if (nS > 0) {
    for (i in seq_len(nS)) {
      forms[[1]] <- c(forms[[1]], as.formula(paste0("S", i, " ~ 1")))
      pars[paste0("S", i)] <- list(list(beta = beta_cov, phi = 1))
    }
  }
  
  # Specify the formula for A given covariates
  ## Add I to the propensity score formula
  if (nI > 0) {
    for (i in seq_len(nI)) {
      forms[[2]] <- update.formula(forms[[2]], paste0("A ~ . + I", i))
    }
  }
  
  ## Add X to the propensity score formula
  if (nX > 0) {
    for (i in seq_len(nX)) {
      forms[[2]] <- update.formula(forms[[2]], paste0("A ~ . + X", i))
    }
  }
  
  # Parameters for copula
  parY <- list()
  parY_names <- c()

  if (nX > 0) {
    parY <- c(parY, rep(list(list(beta = strength_outcome)), nX))
    parY_names <- c(parY_names, paste0("X", seq_len(nX)))
  }
  if (nO > 0) {
    parY <- c(parY, rep(list(list(beta = strength_outcome)), nO))
    parY_names <- c(parY_names, paste0("O", seq_len(nO)))
  }
  if (nI > 0) {
    parY <- c(parY, rep(list(list(beta = 0)), nI))
    parY_names <- c(parY_names, paste0("I", seq_len(nI)))
  }
  if (nS > 0) {
    parY <- c(parY, rep(list(list(beta = 0)), nS))
    parY_names <- c(parY_names, paste0("S", seq_len(nS)))
  }

  names(parY) <- parY_names
  pars$cop <- list(Y = parY)

  
  # Set parameters for A
  pars$A$beta <- c(0, rep(strength_instr, nI), rep(strength_conf, nX))
  if (!binary_intervention) {
    pars$A$phi <- 1
  }
  
  # Set parameters for Y
  pars$Y$beta <- c(0, ate)
  pars$Y$phi <- 1
  
  # Generate data
  df <- rfrugalParam(n = n, formulas = forms, pars = pars, family = fam)
  p <- nX + nI + nO + nS
  
  # Propensity score
  if (binary_intervention) {
    df['propen'] <- plogis(rowSums(c(rep(strength_instr, nI), rep(strength_conf, nX)) * df[, c(1:(nI + nX))]))
    colnames(df) <- c(paste("X", 1:p, sep = ""), 'A', 'y', 'propen')
  } else {
    colnames(df) <- c(paste("X", 1:p, sep = ""), 'A', 'y')
  }
  
  return(df)
}
