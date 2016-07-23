#! usr/bin/python
# coding: utf-8

import os
import math
import numpy as np
from d_profile import profile
from scipy import special
from corr_lda import CorrLDA

#SET PARAMETER
K = 50
repeat = 200
alpha = 50.0 / K 
beta = 0.1
eta = 0.5
lam = 0.5
save_step = 50

os.chdir('./sample/data')
qiita_voc = 'vocab.qiita.txt'
qiita_doc = 'docword.qiita.txt'
tag_voc =  'tag.qiita.txt'
tag_doc =  'doctag.qiita.txt'

lda = CorrLDA(K, alpha, beta, eta, lam)
vdict = lda.make_dict(qiita_voc)
sdict = lda.make_dict(tag_voc)
lda.set_corpus(qiita_doc)
lda.set_supply(tag_doc)

#OUTPUT PARAMETER
Rank = 10

os.chdir('../output')
fwname = "result"
pfile = open("perplexity.txt","w")

print 'start iteration'
for re in range(repeat):
  perplexity = lda.inference()
  wline = str(re + 1) + ' ' + str(perplexity) + "\n"
  pfile.write(wline)
  print wline[:-1]
  if (re + 1) % save_step == 0:
    lda.save(fwname, vdict, re + 1, Rank)
    lda.save_supply(fwname, sdict, re + 1, Rank)
pfile.close()
