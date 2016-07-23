# CorrLDA
This repository is the Python code of the Noisy Correspondence Topic Model proposed by T.Iwata (2013).

# introduction
This repository is the source code of the Noisy Correspondence Topic Model originated from [T.Iwata et al (2013)](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=6193102&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel5%2F69%2F6517838%2F06193102.pdf%3Farnumber%3D6193102).

A probabilistic topic model for analyzing and extracting content-related annotations from noisy annotated discrete data. 

This repository is the solution for the model using the Correspondence-Latent Dirichlet Allocation (corr-LDA).
The inference is based on collapsed Gibbs Sampling.

# how to use
Run a code using sample data easily in python 2.x.

> python run.py

# Modification
The source code is made of a C++ wrapper of Python (Cython).

If you want to modify the source code (corr_lda.pyx), you need to prepare the C++ compiler and run setup.py.

> python setup.py build_ext --inplace

# Dataset
The dataset of the sample are made by the word extraction of the articles of [Qiita](http://qiita.com/) which is Japanese SNS service for the programmers.

These articles attach tags about main sentences, therefore these can utilize for annotations.

The dataset constitution is based on [UCI Machine Learning Repository: Bag of Words Data Set](http://archive.ics.uci.edu/ml/datasets/Bag+of+Words)

# output

While running, the perplexity per an iteration is output to the console.

Example:
> \#iteration, perplexity

>     1 2682.5625869

>     2 2144.99446438

>     ...

You can change the save step for making the result file by changing the 'save_step' value in run.py

### profile output

- set_corpus

When setting 'True' in the 'set\_corpus' function, the profile is output to the console. 

Example:
> \#docID, the number of words that are assigned to the topics in the docID document (list), the number of words in the docID document

>     1 [  0.   0.   5.   3.   0.  15.   0.   0.   0.   3.   1.   8.   0.   0.   0.   6.   2.   0.   3.   9.   0.   0.   7.   0.   0.   6.   2.   0.   0.   2.   2.   0.   0.   0.   0.   2.   2.   0.   1.   0.   2.   2.   0.   0.   1.   4.   0.   3.   0.   3.] 94

>     2 [ 0.  2.  0.  0.  0.  0.  2.  0.  0.  0.  3.  0.  0.  0.  0.  1.  2.  0.   0.  0.  0.  1.  2.  1.  0.  1.  5.  0.  1.  0.  0.  0.  1.  2.  0.  4.   0.  2.  0.  1.  0.  2.  0.  1.  2.  2.  0.  1.  4.  0.] 43

>     ...

- set_supply

When setting 'True' in the 'set\_supply' function, the profile is output to the console. 

Example:
> \#docID, the number of annotations that are assigned to the topics(list) and the number of noisy annotation that are assigned in the docId document, the number of annotation in the docID document

>     1 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.] 0.0 1

>     2 [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.] 0.0 2

>     ...

- inference_prof

When using the 'inference\_prof' function, the profile of inference calculation is output to the console.

Example:
>     <<<---

>          3755177 function calls in 4.620 seconds

> 

>       Ordered by: cumulative time

>       List reduced from 22 to 20 due to restriction <20>

> 

>     ncalls  tottime  percall  cumtime  percall filename:lineno(function)

>     1877571    0.583    0.000    4.593    0.000 _methods.py:31(_sum)

>     1877571    4.010    0.000    4.010    0.000 {method 'reduce' of 'numpy.ufunc' objects}

>         3    0.000    0.000    0.027    0.009 warnings.py:24(_show_warning)

>         3    0.000    0.000    0.027    0.009 warnings.py:36(formatwarning)

>         3    0.000    0.000    0.027    0.009 linecache.py:13(getline)

>         3    0.000    0.000    0.027    0.009 linecache.py:33(getlines)

>         1    0.000    0.000    0.027    0.027 linecache.py:68(updatecache)

>         1    0.027    0.027    0.027    0.027 {method 'readlines' of 'file' objects}

>         3    0.000    0.000    0.000    0.000 {method 'write' of 'file' objects}

>         1    0.000    0.000    0.000    0.000 pstats.py:62(__init__)

>         1    0.000    0.000    0.000    0.000 pstats.py:84(init)

>         1    0.000    0.000    0.000    0.000 pstats.py:106(load_stats)

>         1    0.000    0.000    0.000    0.000 {posix.stat}

>         1    0.000    0.000    0.000    0.000 {open}

>         1    0.000    0.000    0.000    0.000 cProfile.py:90(create_stats)

>         4    0.000    0.000    0.000    0.000 {len}

>         3    0.000    0.000    0.000    0.000 {method 'strip' of 'str' objects}

>         1    0.000    0.000    0.000    0.000 {method 'endswith' of 'str' objects}

>         1    0.000    0.000    0.000    0.000 {isinstance}

>         1    0.000    0.000    0.000    0.000 {hasattr}

> 

> 

> 

>     --->>>
