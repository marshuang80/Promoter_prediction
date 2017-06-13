#!/user/bin/env python
'''
data_generator.py: Generate batches of data for the neural network

Authorship information:
    __author__ = "Mars Huang"
    __email__ = "marshuang80@gmai.com:
    __status__ = "complete"
'''
import pickle
import random
import numpy as np
import math

def load_data():
    '''Load data from old data and new data. Match rpkm from old file to one-hot
    encoding of new data.

    Input files:
        "onehot500.pickle": one hot ecnoded promoter sequence from 0 to -500
                            (previously defined promoters), along with rpkm
                            values of the gene.
        onehot500.-450+50.txt: one hot encoded promoter sequence from +50 to
                               -450.
    '''
    # Load data
    old_promoters = pickle.load(open("onehot500.pickle",'rb'))
    new_promoters = pickle.load(open("onehot500bp.-450+50.txt",'rb'))

    # Make promoter dictionary
    promo_dict = {a[0]:[a[2],None] for a in new_promoters}

    # Match rpkm values from old file
    for p in old_promoters:
        if p[0] in promo_dict:
            promo_dict[p[0]][1] = float(p[2])
        else:
            continue

    # Extract data and generate features (x) and labels(y)
    promoters = [[v[0],v[1]] for k,v in promo_dict.items() if v[1] != None]
    x =[a[0].T for a in promoters]
    y = [a[1] for a in promoters]
    return x, y


def even_distribution(x,y):
    '''Generate three different range of data to sample from

    Args:
        x (np.array): one-hot encoding of the gene
        y (float): rpkm values of the gene

    Return:
        low (np.array): data from the lower range
        mid (np.array): data from the mid range
        high (np.array): data from the upper range
     '''
    # Take a log of rpkm to normalize data
    data = [(a,math.log(b)) for a,b in zip(x,y)]

    # Split the data input 3 ranges
    low = np.array([a for a in data if a[1] <= 1.0])
    mid = np.array([a for a in data if (a[1] > 1.0 and a[1] <2)])
    high = np.array([a for a in data if a[1] >=2])
    return low, mid, high


def shuffle_data(x,y):
    '''Shuffle data

    Args:
        x (np.array): one-hot encoding of the gene
        y (float): rpkm values of the gene

    Return:
        list(x[idx]) (list): list of shuffled input
        list(y[idx]) (list): list of shuffled labels
    '''
    x = np.array(x)
    y = np.array(y)
    np.random.seed(7)
    idx = np.random.permutation(np.arange(len(y)))
    return list(x[idx]), list(y[idx])


def batch(l, m ,h,size):
    '''Generate a batch of data for training the neural network

    Args:
        l (np.array): data from the lower range
        m (np.array): data from the mid range
        h (np.array): data from the upper range
        size (int): the size of the training batch size

    Return:
        x (list): batch of training data
        y (list): batch of training label
    '''
    # Gernate random indexs to extract from training data
    np.random.seed(7)
    idx = np.random.permutation(np.arange(size//3))
    data = np.concatenate([l[idx],h[idx],m[idx]])
    x = list(data[:,0])
    y = list(data[:,1])
    y = [[a] for a in y]
    return x,y


def generate_data():
    '''Generate evenly distributed data for neural network

    Return:
        l (np.array): data from the lower range
        m (np.array): data from the mid range
        h (np.array): data from the upper range
    '''
    x,y = load_data()
    l,m,h = even_distribution(x,y)
    return l,m,h

