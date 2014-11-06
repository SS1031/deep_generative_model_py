#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
t-distributed Stochastic Neighbor Embedding (t-SNE)
'''
import time
import cPickle
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import offsetbox

# from sklearn.preprocessing import Binarizer
from sklearn.manifold import TSNE


def load_data(file_name):
    '''
    load dataset

    @param  file_name   path to file

    @return             dataset
    '''
    print '... loading data -> ' + file_name
    pkl_file = open(file_name, 'r')
    dataset = cPickle.load(pkl_file)
    pkl_file.close()
    return dataset


def plot_embedding(X, y, title=None):
    '''
    plotting dataset to sne cordinates

    @type   X       list
    @param  X       cordinates that calulated by sne
    @type   y       list
    @param  y       label of dataset
    @type   title   string
    @param  title   title of the plot
    '''
    print '... plotting'
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    print X.shape[0]
    print len(y)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 20.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()


def plot_t_sne(t, X, y):
    print '... transforming'
    model = TSNE(n_components=t, random_state=0)
    X_tsne = model.fit_transform(X)
    plot_embedding(X_tsne, y)

if __name__ == '__main__':

    dataset = load_data('../result/hashed_20NG.pkl')

    X = dataset[0][0:1000]
    # if you want to binarize hashed vecs, you can use below code
    # X = Binarizer(threshold=0.1).fit_transform(X)
    y = dataset[1][0:1000]

    t0 = time.clock()
    plot_t_sne(t=2, X=X, y=y)

    print 'plotting time = ',
    print (time.clock() - t0)
