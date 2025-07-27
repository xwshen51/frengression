# Frengression
This repository contains code for our article, [Frugal, Flexible, Faithful: Causal Data Simulation via Frengression](http://arxiv.org/abs/2503.15989).

# Abstract
Machine learning has revitalized causal inference by combining flexible models and principled estimators, yet robust benchmarking and evaluation remain challenging with real-world data. In this work, we introduce frengression, a deep generative realization of the frugal parameterization that models the joint distribution of covariates, treatments and outcomes around the causal margin of interest. Frengression provides accurate estimation and flexible, faithful simulation of multivariate, time-varying data; it also enables direct sampling from user-specified interventional distributions. Model consistency and extrapolation guarantees are established, with validation on real-world clinical trial data demonstrating frengression’s practical utility. We envision this framework sparking new research into generative approaches for causal margin modelling.

# Structure
Frengression model is stored at ```frengression.py```.  To replicate the experiments in the paper, please check the ```paper_exp``` folder for the jupyter notebooks.

# Requirements
The R packages, ```causl``` and ```survivl``` need to be installed to run synthetic experiments. You can install the package via:
1. Install ```devtools```
   ```
   install.packages("devtools")
   library(devtools)
   ```
2. Install ```causl``` and ```survivl```
   ```
   install_github("rje42/causl")

   install_github("rje42/survivl")
   ```
Details of the packages can be found in [https://github.com/rje42/causl](https://github.com/rje42/causl) and [https://github.com/rje42/survivl](https://github.com/rje42/survivl).