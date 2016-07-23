# CorrLDA
This repository is the Python code of the Noisy Correspondence Topic Model proposed by T.Iwata (2013).

# introduction
This repository is the source code of the Noisy Correspondence Topic Model proposed by [T.Iwata et. al](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=6193102&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel5%2F69%2F6517838%2F06193102.pdf%3Farnumber%3D6193102).

A probabilistic topic model for analyzing and extracting content-related annotations from noisy annotated discrete data. 

This repository is the solution for the model using the Correspondence-Latent Dirichlet Allocation (corr-LDA).
The inference is based on collapsed Gibbs Sampling.

# How to use
You can run for this sample data easily in python 2.x.

> :python run.py

# Modification
This source code is made of a C++ wrapper of Python (Cython).
if you want to modify the source code (corr_lda.pyx), need to prepare the C++ compiler and run a setup.py.

> :python setup.py build_ext --inplace
