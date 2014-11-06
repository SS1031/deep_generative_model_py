#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import gzip
import cPickle
import os

DATA_DIR = '../data/20news-bydate/'
OUTPUT_DIR = '../data/'


def load_category():
    print '... load categories'
    category_list = []
    fp = open(DATA_DIR + 'train.map')  # test.map is same data
    for line in fp:
        line = line.rstrip()
        category_list.append(line.split()[0])
    fp.close()
    return category_list


def load_vocabulary():
    print '... load vocabularies'
    vocabulary_list = []
    fp = open(DATA_DIR + 'vocabulary.txt')
    for line in fp:
        line = line.rstrip()
        vocabulary_list.append(line)
    fp.close()
    return vocabulary_list


def load_label():
    print '... load labels'
    label_list = []
    fp = open(DATA_DIR + 'train.label')
    for line in fp:
        line = line.rstrip()
        # original data label start at 1, so use -1 index
        label_list.append(int(line) - 1)
    fp.close()

    fp = open(DATA_DIR + 'test.label')
    for line in fp:
        line = line.rstrip()
        # original data label start at 1, so use -1 index
        label_list.append(int(line) - 1)
    fp.close()

    return label_list


if __name__ == '__main__':

    categoty_list = load_category()
    vocabulary_list = load_vocabulary()
    label_list = load_label()

    # create new numpy array for dataset vectors
    data_vecs = np.zeros((len(label_list), len(vocabulary_list)))

    print '... load train data'
    fp = open(DATA_DIR + 'train.data')
    train_num = 0
    for line in fp:
        line = line.strip()
        temp = line.split()
        docidx, wordidx, count = int(temp[0]), int(temp[1]), int(temp[2])
        # original data index begin with 1, so use -1 index
        data_vecs[docidx-1][wordidx-1] = count
    fp.close()
    train_num = docidx

    print '... load test data'
    fp = open(DATA_DIR + 'test.data')
    for line in fp:
        line = line.strip()
        temp = line.split()
        docidx, wordidx, count = int(temp[0]), int(temp[1]), int(temp[2])
        # original data index begin with 1, so use index - 1
        data_vecs[docidx-1 + train_num][wordidx-1] = count
    fp.close()

    for_sort_data_vecs = np.sum(data_vecs, axis=0)

    # top 2000 of appearance frequency
    top2000_indecies = for_sort_data_vecs.argsort()[-2000:][::-1]
    data_vecs = data_vecs[:, top2000_indecies]

    for vec in data_vecs:
        vec /= np.amax(vec)

    # devide to 'train', 'validata', 'test' dataset
    dataset = [
        [data_vecs[0:8314], label_list[0:8314]],                    # train set
        [data_vecs[8314:train_num], label_list[8314:train_num]],    # valid set
        [data_vecs[train_num:len(label_list)],
         label_list[train_num:len(label_list)]]                     # test set
    ]

    # setup ouput directory
    os.chdir(OUTPUT_DIR)

    filename = 'preprocessed_20NG.pkl.gz'
    print '... save -> ' + OUTPUT_DIR + filename
    gz_pkl_file = gzip.GzipFile(filename, 'w')
    gz_pkl_file.write(cPickle.dumps(dataset))
    gz_pkl_file.close()
