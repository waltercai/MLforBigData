THRESHOLD = 0.0001

import sys
import numpy as np
from scipy.stats import multivariate_normal as mn
from scipy.linalg import norm
from scipy.misc import logsumexp
import pprint
import time
import copy
import random

import matplotlib.pyplot as plt
import seaborn as sea

def e_step(docs, log_prior, mu, sigma):
    num_docs = len(docs)
    K = len(mu)
    log_resp = np.zeros(shape=[num_docs, K]);
    for i in range(num_docs):
        # fill in the raw values for the row corresponding to document i
        for k in range(K):
            log_prob = mn.logpdf(x=docs[i,:],
                                 mean=mu[k,:],
                                 cov=sigma[k,:,:])
            log_resp[i,k] = log_prior[k] + log_prob
        # normalize each row
        log_row_sum = logsumexp(log_resp[i,:])
        log_resp[i,:] -= log_row_sum
    
    return log_resp

def m_step(log_resp, docs):
    num_docs = len(docs)
    K = len(log_resp[0,:])
    dim = len(docs[0,:])
    
    log_prior = np.zeros(K)
    mu = np.zeros(shape=[K,dim])
    sigma = np.zeros(shape=[K, dim, dim])
    
    for k in range(K):
        col_sum = np.sum(np.exp(log_resp[:,k]))
        
        # recalcuate prior
        log_prior[k] = np.log(col_sum / num_docs)
        
        # recalculate mu
        for i in range(num_docs):
            mu[k,:] += np.exp(log_resp[i,k]) * docs[i,:]
        mu[k,:] = mu[k,:] / col_sum
    
    for k in range(K):
        col_sum = np.sum(np.exp(log_resp[:,k]))
        
        # recalculate sigma
        for i in range(num_docs):
            sigma[k,:,:] += np.exp(log_resp[i,k]) * np.outer(docs[i,:]-mu[k,:], docs[i,:]-mu[k,:])
        sigma[k,:,:] /= col_sum
        
    lambduh = 0.2
    sigma = (1 - lambduh) * sigma + lambduh * np.identity(dim)
        
    return [log_prior, mu, sigma]
        
def get_ll(log_resp, docs, log_prior, mu, sigma):
    num_docs = len(docs)
    K = len(mu)
    
    ll = 0.0
    
    for i in range(num_docs):
        for k in range(K):
            inner = log_prior[k] + mn.logpdf(x=docs[i,:],
                                              mean=mu[k,:],
                                              cov=sigma[k,:,:])
            ll += np.exp(log_resp[i,k]) * inner
    
    return ll

labels = []
num_docs = 0
num_terms = 0
K=5
docs_dict = []

with open("/Users/waltercai/Documents/cse547/hw2/bbc_data/bbc.mtx") as mtx:
    line_count = 0
    for line in mtx:
        line_count+=1
        if line_count == 1:
            pass
        elif line_count == 2:
            line_split = line.split(" ")
            num_terms = int(line_split[0])
            num_docs = int(line_split[1])
            
            for i in range(num_docs):
                docs_dict.append({})
            len(docs_dict)
        else:
            line_split = line.split(" ")
            # change to 0 indexing
            term = int(line_split[0]) - 1
            doc = int(line_split[1]) - 1
            freq = float(line_split[2])
            
            docs_dict[doc][term] = freq

# generate tf matrix
tf = np.zeros(shape=[num_docs, num_terms])
for i in range(num_docs):
    for k in docs_dict[i].keys():
        tf[i,k] = docs_dict[i][k]
    tf[i,:] = tf[i,:] / np.max(tf[i,:])

# generate idf vector
idf = np.zeros(num_terms)
for t in range(num_terms):
    idf[t] = np.log((num_docs + 0.0) / np.count_nonzero(tf[:,t]))
    
for t in range(num_terms):
    tf[:,t] *= idf[t]

# generate tfidf matrix
tfidf = tf

# get true classes
true_label = []
with open("/Users/waltercai/Documents/cse547/hw2/bbc_data/bbc.classes") as classes:
    for line in classes:
        line_split = line.split(" ")
        true_label.append(int(line_split[1]))
true_label = np.array(true_label)

# get cluster sizes
true_clust_size = np.zeros(K)
for t in range(num_docs):
    true_clust_size[true_label[t]] += 1

# get terms
term_strings = []
with open("/Users/waltercai/Documents/cse547/hw2/bbc_data/bbc.terms") as terms:
    for line in terms:
        term_strings.append(line[0:-1])

# generate term-doc-frequency matrix
tdf = np.zeros(shape=[num_terms,K])
for d in range(num_docs):
    for t in range(num_terms):
        tdf[t, true_label[d]] += tfidf[d,t]
for k in range(K):
    tdf[:,k] /= true_clust_size[k]

print("top 5 avg tfidf terms for each cluster:")
for k in range(K):
    arr = tdf[:,k]
    print [term_strings[i] for i in list(arr.argsort()[-5:][::-1])]

# initialize mu
mu = np.zeros(shape=[K, num_terms])
with open("/Users/waltercai/Documents/cse547/hw2/bbc_data/bbc.centers") as centers:
    line_count = 0
    for line in centers:
        line_split = line.split(" ")
        for t in range(num_terms):
            mu[line_count, t] = float(line_split[t])
        line_count+=1
log_prior = np.log(np.zeros(K) + 1.0/K)
log_resp = np.zeros(shape=[num_docs, K]);

sigma = np.zeros(shape=[K, num_terms, num_terms])
for k in range(K):
    sigma[k,:,:] = np.identity(num_terms)

old_ll = 0.0

lls = []
loss = []
for i in range(5):
    print("iteration: {}".format(i+1))
    log_resp = e_step(docs=tfidf, log_prior=log_prior, mu=mu, sigma=sigma)
    [prior, mu, sigma] = m_step(log_resp=log_resp, docs=tfidf)
    
    guess_list = []
    for i in range(num_docs):
        k = np.argmax(np.exp(log_resp[i,:]))
        guess_list.append(k)
    loss.append(np.count_nonzero(np.array(guess_list) - np.array(true_label)))
    
    new_ll = get_ll(log_resp=log_resp, docs=tfidf, log_prior=log_prior, mu=mu, sigma=sigma)
    lls.append(new_ll)

# print("mu(s): {}\n".format(mu))
# print("sigma(s): {}\n".format(sigma))
guess = {}
for k in range(K):
    guess[k] = []
for i in range(num_docs):
    k = np.argmax(np.exp(log_resp[i,:]))
    guess[k].append(i)
# for k in range(K):
#     print("guess cluster {}: {}\n".format(k, guess[k]))
print("log likelihoods: {}".format(lls))
print("0/1 Loss: {}".format(loss))
print("num documents: {}".format(num_docs))
