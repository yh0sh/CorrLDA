# distutils: language=c++
# -*- coding: utf-8 -*-

import os
import math
from scipy import special 
from d_profile import profile

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

DTYPE = np.int
FTYPE = np.float
ctypedef np.int_t DTYPE_t
ctypedef np.float_t FTYPE_t

cdef class CorrLDA:
  # MAIN VARIABLES
  cdef int K, D, V
  cdef double alpha, beta
  cdef np.ndarray Nk
  cdef np.ndarray Ndk
  cdef np.ndarray Nkv
  cdef np.ndarray Nd
  cdef np.ndarray zdn # topic sequence for word n in document d
  cdef np.ndarray vdn # vocaburary sequence for word n in document d
  # SUPPLY INFORMATION VARIABLES
  cdef int S
  cdef double M0, M1
  cdef double gamma, eta, lam
  cdef np.ndarray Mk
  cdef np.ndarray Mdk
  cdef np.ndarray M0s
  cdef np.ndarray Mks
  cdef np.ndarray Md
  cdef np.ndarray xdm # topic sequence for supply m in document d
  cdef np.ndarray sdm # vocaburary sequence for supply m in document d
  cdef np.ndarray rdm # relevance sequence for supply m in document d

  def __init__(self, K, alpha, beta, eta, lam):
    self.K = K
    self.alpha = alpha
    self.beta = beta
    self.eta = eta
    self.lam = lam
    self.gamma = np.random.beta(self.lam, self.lam)

  # make document dictionary with a key of document id
  def make_dict(self, vfile):
    vdict = {}
    fvoc = open(vfile)
    for num,line in enumerate(fvoc):
      vdict[num] = line[:-1]
    fvoc.close()
    return vdict

  # count up Nk, Ndk, Nkv, and Nd
  def set_corpus(self, dfile, prof = False):
    self.Nk = np.zeros(self.K,dtype=FTYPE)
    fdoc = open(dfile)
    pre_doc = 0
    cdef vector[int] zdn
    cdef vector[int] vdn
    cdef int count
    for num, line in enumerate(fdoc):
      if num == 0:
        self.D = int(line[:-1])
        self.Ndk = np.zeros([self.D, self.K],dtype=FTYPE)
        self.Nd = np.zeros(self.D,dtype=DTYPE)
        continue
      elif num == 1:
        self.V = int(line[:-1])
        self.Nkv = np.zeros([self.K, self.V],dtype=FTYPE)
        continue
      elif num == 2:
        continue
      else:
        d, v, c = line[:-1].split(' ')
        k = np.random.randint(self.K,size=1)[0]
        doc = int(d) - 1
        voc = int(v) - 1
        count = int(c)
        self.Ndk[doc, k] += count
        self.Nkv[k, voc] += count
        self.Nk[k] += count
        self.Nd[doc] += count
        for c in range(count):
          vdn.push_back(voc)
          zdn.push_back(k)
        if pre_doc != doc and prof:
          print pre_doc+1, self.Ndk[pre_doc], self.Nd[pre_doc]
          pre_doc = doc
    if prof:
      print pre_doc+1, self.Ndk[pre_doc], self.Nd[pre_doc]
    fdoc.close()
    self.Nk += self.beta * self.V
    #self.Ndk += self.alpha
    self.Nkv += self.beta
    self.zdn = np.array(list(zdn))
    self.vdn = np.array(list(vdn))
    return 0

  # count up M0, M1, Mk, Mdk, M0s, Mks, and Md
  def set_supply(self, dfile, prof = False):
    self.M0 = 0
    self.M1 = 0
    self.Mk = np.zeros(self.K + 1,dtype=FTYPE) # K + 1 includes the index of noisy topic
    self.Mdk = np.zeros([self.D, self.K],dtype=FTYPE)
    self.Md = np.zeros(self.D,dtype=DTYPE) # for numbering iteration only, not using calculation
    fdoc = open(dfile)
    pre_doc = 0
    cdef vector[int] xdm
    cdef vector[int] sdm
    cdef vector[int] rdm
    cdef int count, num_noisy
    for num, line in enumerate(fdoc):
      if num == 0 or num == 2:
        continue
      elif num == 1:
        self.S = int(line[:-1])
        self.Mks = np.zeros([self.K, self.S],dtype=FTYPE)
        self.M0s = np.zeros(self.S,dtype=FTYPE)
        continue
      else:
        r = np.random.binomial(1, self.gamma) # setting relevance randomly
        d, s, c = line[:-1].split(' ')
        doc = int(d) - 1
        supply = int(s) - 1
        count = int(c)
        if r == 1: # relevance
          k = np.random.multinomial(1,self.Ndk[doc] / self.Ndk[doc].sum()).argmax()
          self.Mdk[doc, k] += count
          self.Mks[k, supply] += count
          self.M1 += count
        else: # irrelevance
          k = self.K  
          self.M0s[supply] += count
          self.M0 += count
          num_noisy += count
        self.Mk[k] += count
        self.Md[doc] += count
        for c in range(count):
          sdm.push_back(supply)
          xdm.push_back(k)
          rdm.push_back(r)
        # if prof is true, print out below:
        # doc-id, word count of all topics(array), that of noisy topic, total count of words in the document with doc-id
        if pre_doc != doc and prof:
          print pre_doc+1, self.Mdk[pre_doc], num_noisy, self.Md[pre_doc]
          pre_doc = doc
          num_noisy = 0
    if prof:
      print pre_doc+1, self.Mdk[pre_doc], num_noisy, self.Md[pre_doc]
    fdoc.close()
    self.M0 += self.eta
    self.M1 += self.eta
    self.Mk += self.gamma * self.S
    self.Mks += self.gamma
    self.M0s += self.gamma
    self.xdm = np.array(list(xdm))
    self.sdm = np.array(list(sdm))
    self.rdm = np.array(list(rdm))
    return 0

  # if prof is true, print out the profile of the inference
  @profile
  def inference_prof(self):
    return self.inference()

  def inference(self):
    # MAIN VARIABLES
    cdef int t, v, new_t
    cdef int idx = 0 # index of topic for main
    cdef np.ndarray[FTYPE_t, ndim=1] theta = np.zeros(self.K,dtype=FTYPE)
    cdef np.ndarray[FTYPE_t, ndim=1] phi = np.zeros(self.K,dtype=FTYPE)
    cdef np.ndarray[FTYPE_t, ndim=1] p_z = np.zeros(self.K,dtype=FTYPE)
    # SUPPLY INFORMATION VARIABLES
    cdef int k, s, new_ts, ktemp = 0
    cdef int idxs = 0 # index of topic for supply information
    cdef double p_r0
    cdef np.ndarray[FTYPE_t, ndim=1] p_r1 = np.zeros(self.K,dtype=FTYPE)
    cdef np.ndarray[FTYPE_t, ndim=1] p_y = np.zeros(self.K,dtype=FTYPE)

    for d in range(len(self.Nd)):
      # MAIN ITERATION
      for n in range(self.Nd[d]):
        t = self.zdn[idx]
        v = self.vdn[idx]
        # REDUCE COUNT OF OLD TOPIC
        self.Ndk[d][t] -= 1
        self.Nkv[t][v] -= 1
        self.Nk[t] -= 1
        # INFERENCE
        theta = self.Ndk[d] + self.alpha
        phi = self.Nkv[:,v] / self.Nk
        p_z = theta * phi * ((self.Ndk[d] + 1.0) / self.Ndk[d])**self.Mdk[d]
        new_t = np.random.multinomial(1,p_z / p_z.sum()).argmax() # decide new topic as categorical dist
        # COUNT ADD TO NEW TOPIC
        self.zdn[idx] = new_t
        self.Ndk[d][new_t] += 1
        self.Nkv[new_t][v] += 1
        self.Nk[new_t] += 1
        idx += 1

      # ITERATION FOR SUPPLY INFOMATION
      for n_s in range(self.Md[d]):
        k = self.xdm[idxs] # old topic
        s = self.sdm[idxs]
        # REDUCE COUNT OF OLD TOPIC
        self.Mk[k] -= 1
        if self.rdm[idxs] == 1:
          self.Mdk[d, k] -= 1
          self.Mks[k, s] -= 1
          self.M1 -= 1
        else:
          self.M0s[s] -= 1
          self.M0 -= 1
        p_r1 = self.M1 * self.Mks[:, s] / self.Mk[:self.K]
        p_r0 = self.M0 * self.M0s[s] / self.Mk[self.K]
        r = np.random.binomial(1, p_r1.sum() / (p_r0 + p_r1.sum()))
        # COUNT ADD TO NEW TOPIC
        self.rdm[idxs] = r # replace to new rdm = {0,1}
        if r == 1:
          p_y = self.Ndk[d] * (self.Mks[:,s] + self.gamma) / self.Mk[k]
          new_ts = np.random.multinomial(1,p_y / p_y.sum()).argmax()
          self.xdm[idxs] = new_ts # replace to new topic
          self.Mdk[d, new_ts] += 1
          self.Mks[new_ts, s] += 1
          self.M1 += 1
        else:
          self.xdm[idxs] = self.K # replace to noisy topic
          self.M0s[s] += 1
          self.M0 += 1
        self.Mk[self.xdm[idxs]] += 1
        idxs += 1
    cdef double perplexity = self.__calc_perplexity()
    return perplexity

  # calculation perplexity per one iteration
  def __calc_perplexity(self):
    cdef int s, N, idx = 0
    cdef double log_per = 0.0
    cdef double Kalpha = self.K * self.alpha
    cdef np.ndarray[FTYPE_t, ndim=1] theta = np.zeros(self.K,dtype=FTYPE)
    cdef np.ndarray[FTYPE_t, ndim=2] phi = self.Nkv / self.Nk[:,np.newaxis]
    for d in range(len(self.Nd)):
      theta = self.Ndk[d,:] / (self.Nd[d] + Kalpha) * ((self.Ndk[d] + 1.0) / self.Ndk[d])**self.Mdk[d]
      for v in range(self.Nd[d]):
        v = self.vdn[idx]
        log_per -= np.log(np.inner(phi[:,v], theta))
        idx += 1
    N = idx + 1
    return np.exp(log_per / N)

  # output vocaburaries with higher probability in one topic
  def __output(self, output, res, Rank, prof = False):
    fw = open(output,'w')
    for k in res.keys():
      topic_no = 'topic ' + str(k) + '\n'
      fw.write(topic_no)
      if prof:
        print topic_no[:-1]
      rank = 0
      for v, prob in sorted(res[k].items(), key=lambda x:x[1], reverse=True):
        voc_prob = str(v) + ' : ' + str(prob) + '\n'
        fw.write(voc_prob)
        if prof:
          print voc_prob[:-1]
        rank += 1
        if rank == Rank:
          break
      fw.write('\n')
    fw.close()
  
  # save result
  def save(self, fname, vdict, num, Rank, prof = False):
    output = fname + str(num) + '.txt'
    res = {}
    
    print 'writing file ' + output + '...'
    for k in range(self.K):
      res[k] = {}
      for v in range(self.V):
        prob = 1.0 * self.Nkv[k][v] / self.Nkv[k].sum()
        if prob == 0.0:
          continue
        res[k][vdict[v]] = prob
    self.__output(output, res, Rank ,prof)

  # save result of supply
  def save_supply(self, fname, vdict, num, Rank, prof = False):
    output = fname + str(num) + '.supply.txt'
    res = {}
    
    print 'writing file ' + output + '...'
    for k in range(self.K):
      res[k] = {}
      for s in range(self.S):
        prob = 1.0 * self.Mks[k][s] / self.Mks[k].sum()
        if prob == 0.0:
          continue
        res[k][vdict[s]] = prob
    res[self.K] = {}
    for s in range(self.S):
      prob = 1.0 * self.M0s[s] / self.M0s.sum()
      if prob == 0.0:
        continue
      res[self.K][vdict[s]] = prob
    self.__output(output, res, Rank ,prof)
