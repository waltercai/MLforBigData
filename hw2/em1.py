THRESHOLD = 0.0001

import sys
import numpy as np
from scipy.stats import multivariate_normal as mn
from scipy.linalg import norm
import pprint
import time
import copy
import random

import matplotlib.pyplot as plt
import seaborn as sea

def kpp_init(docs, mu):
    num_docs = len(docs)
    K = len(mu)
    index = np.random.choice(num_docs)
    mu[0,:] = docs[index,:]
    
    for k in range(1,K):
        min_dist_array= []
        for i in range(num_docs):
            min_dist = norm(docs[i,:] - mu[0,:])
            for j in range(i,k):
                new_dist = norm(docs[i,:] - mu[j,:])
                if new_dist < min_dist:
                    min_dist = new_dist
            min_dist_array.append(min_dist)
        index = np.random.choice(num_docs, p=min_dist_array/np.sum(min_dist_array))
        mu[k,:] = docs[index,:]

def e_step(docs, prior, mu, sigma):
    num_docs = len(docs)
    K = len(mu)
    resp = np.zeros(shape=[num_docs, K]);
    for i in range(num_docs):
        # fill in the raw values for the row corresponding to document i
        for k in range(K):
            prob = mn.pdf(x=docs[i,:],
                          mean=mu[k,:],
                          cov=sigma[k,:,:])
            resp[i,k] = prior[k] * prob
        # normalize each row
        row_sum = np.sum(resp[i,:])
        resp[i,:] /= row_sum
    
    return resp

def m_step(resp, docs):
    num_docs = len(docs)
    K = len(resp[0,:])
    dim = len(docs[0,:])
    
    prior = np.zeros(K)
    mu = np.zeros(shape=[K,dim])
    sigma = np.zeros(shape=[K, dim, dim])
    
    for k in range(K):
        col_sum = np.sum(resp[:,k])
        
        # recalcuate prior
        prior[k] = col_sum / num_docs
        
        # recalculate mu
        for i in range(num_docs):
            mu[k,:] += resp[i,k] * docs[i,:]
        mu[k,:] = mu[k,:] / col_sum
    
    for k in range(K):
        col_sum = np.sum(resp[:,k])
        
        # recalculate sigma
        for i in range(num_docs):
            sigma[k,:,:] += resp[i,k] * np.outer(docs[i,:]-mu[k,:], docs[i,:]-mu[k,:])
        sigma[k,:,:] /= col_sum
        
    return [prior, mu, sigma]
        
def get_ll(resp, docs, prior, mu, sigma):
    num_docs = len(docs)
    K = len(mu)
    
    ll = 0.0
    
    for i in range(num_docs):
        for k in range(K):
            inner = np.log(prior[k]) + mn.logpdf(x=docs[i,:],
                                              mean=mu[k,:],
                                              cov=sigma[k,:,:])
            ll += resp[i,k] * inner
    
    return ll

labels = []
num_docs = 0
dim = 0
K=3
docs_dict = []

with open("/Users/waltercai/Documents/cse547/hw2/2DGaussianMixture.csv") as f:
    first_line = True
    for line in f:
        if first_line:
            dim = line.count(",")
            first_line = False
        else:
            line_split = line.split(",")
            labels.append(int(line_split[0]))
            row = {}
            for i in range(dim):
                row[i] = float(line_split[i+1])
            docs_dict.append(row)
    num_docs = len(docs_dict)

docs = np.zeros(shape=[num_docs, dim])
for i in range(num_docs):
    for k in docs_dict[i].keys():
        docs[i,k] = docs_dict[i][k]

prior = np.zeros(K) + 1.0/K
resp = np.zeros(shape=[num_docs, K]);

mu = np.zeros(shape=[K, dim])
sigma = np.zeros(shape=[K, dim, dim])
for k in range(K):
    sigma[k,:,:] = np.identity(dim)


diff = THRESHOLD + 1.0
old_ll = 0.0
kpp_init(docs, mu)

iter_count = 0
lls = []
while diff > THRESHOLD:
# for i in range(500):
    iter_count+=1
    
    resp = e_step(docs=docs, prior=prior, mu=mu, sigma=sigma)
    [prior, mu, sigma] = m_step(resp=resp, docs=docs)
    
    new_ll = get_ll(resp=resp, docs=docs, prior=prior, mu=mu, sigma=sigma)
    lls.append(new_ll)
    diff = np.abs((old_ll - new_ll)/new_ll)
    old_ll = new_ll

print("iterations: {}\n".format(iter_count))
print("mu(s): {}\n".format(mu))
print("sigma(s): {}\n".format(sigma))
guess = {}
for k in range(K):
    guess[k] = []
for i in range(num_docs):
    k = np.argmax(resp[i,:])
    guess[k].append(i)
for k in range(K):
    print("guess cluster {}: {}\n".format(k, guess[k]))
print("log likelihoods: {}".format(lls))


