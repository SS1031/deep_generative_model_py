#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import cPickle
# import gzip
import os
import time
import sys
import numpy

# import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils import load_data
from deep_generative_model import DGM
from t_sne import plot_t_sne

OUTPUT_DIR = '../result/'

'''
   _____                            __  _
  / ___/___  ____ ___  ____ _____  / /_(_)____
  \__ \/ _ \/ __ `__ \/ __ `/ __ \/ __/ / ___/
 ___/ /  __/ / / / / / /_/ / / / / /_/ / /__
/____/\___/_/ /_/ /_/\__,_/_/ /_/\__/_/\___/
                        __  __           __    _
                       / / / /___ ______/ /_  (_)___  ____ _
                      / /_/ / __ `/ ___/ __ \/ / __ \/ __ `/
                     / __  / /_/ (__  ) / / / / / / / /_/ /
                    /_/ /_/\__,_/____/_/ /_/_/_/ /_/\__, /
                                                   /____/

semantic hashing is document feature extraction methods which use a deep
neural network called deep generative model.

reference: http://www.cs.toronto.edu/~rsalakhu/papers/semantic_final.pdf
'''


def trainig_dgm(pretraining_lr=0.1, pretraining_epochs=10,
                finetuning_lr=0.1, finetuning_epochs=10,
                k=1, batch_size=5, datasets=None):
    '''
    @type   pretraining_lr      float
    @param  pretraining_lr      learning rate for pretraining
    @type   pretraining_epochs  int
    @param  pretraining_epochs  pretrain learning epochs
    @type   finetuning_epochs   int
    @param  finetuning_epochs   finetune learning epochs
    @type   k                   int
    @param  k                   for CD-k
    @type   dataset             list of theano.tensor.TensorType
    @param  dataset             'train', 'valid', 'test' datasets
    @type   batch_size          int
    @param  batch_size          size of batch traning

    @return                     trained deep generative model network
    '''

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # define symbolic variable for input
    x = T.matrix('x')

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    input_size = train_set_x.get_value(borrow=True).shape[1]

    print 'input data dimenson = ',
    print input_size

    # init DGM
    dgm = DGM(numpy_rng=rng, theano_rng=theano_rng,
              input=x, input_size=input_size,
              hidden_layers_sizes=[500, 500, 128])

    #########################
    # PRETRAINING THE MODEL #
    #########################
    # calc minibatch n
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    print '... getting the pretraining functions'
    pretraining_fns = dgm.get_pretrain_fns(train_set_x=train_set_x,
                                           batch_size=batch_size, k=k)

    print '... pre-training the model'
    start_time = time.clock()
    # Pre-train layer-wise
    for i in xrange(dgm.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](
                    index=batch_index,
                    lr=pretraining_lr
                    ))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################
    # TODO: add gaussian noise to input for finetuning
    print '... building finetuning model'
    train_fn, validate_model =\
        dgm.build_finetune_functions(datasets=datasets,
                                     batch_size=batch_size,
                                     learning_rate=finetuning_lr)

    # initialize for finetuning
    # TODO: use validation frequency proper
    patience = 4 * n_train_batches
    validation_frequency = min(n_train_batches, patience / 2)
    best_validation_loss = numpy.inf

    start_time = time.clock()

    print '... Start finetuning'
    for epoch in xrange(finetuning_epochs):
        c = []
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # TODO: set validation frequency, now validate all epochs
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation cost %f ' %
                      (epoch, minibatch_index + 1,
                       n_train_batches, this_validation_loss))

                if this_validation_loss < best_validation_loss:
                    # TODO: use the best_params
                    # save the best validation costs and params
                    best_validation_loss = this_validation_loss
                    # best_params = dgm.params

        print ('Training epoch %d, cost ' %
               epoch, numpy.mean(minibatch_avg_cost))

    end_time = time.clock()
    training_time = (end_time - start_time)
    print training_time

    return dgm


def semantic_hashing(dgm, input_dataset):
    '''
    hashing to semantic address by DGM
    return and save hashed vecs

    @type   dgm             DGM class
    @param  dgm             trained dgm
    @type   input_dataset   list of theano.tensor.TensorType
    @param  input_dataset   dataset which contain input data and label

    @return vectors which hashed input
    '''
    print '... hashing'
    input = input_dataset[0]
    label = input_dataset[1]
    hashed_vecs = dgm.encode(input)
    pkl_f_name = 'hashed_20NG.pkl'
    print 'save -> ' + OUTPUT_DIR + pkl_f_name
    pkl_file = open(OUTPUT_DIR + pkl_f_name, 'w')
    # dump with cPickle
    save_data = [hashed_vecs.eval(), label.eval()]
    cPickle.dump(save_data, pkl_file)
    pkl_file.close()
    return hashed_vecs


if __name__ == '__main__':
    datasets = load_data('preprocessed_20NG.pkl.gz')

    trained_dgm = trainig_dgm(pretraining_epochs=1,
                              finetuning_epochs=1,
                              datasets=datasets)

    hashed_vecs = semantic_hashing(trained_dgm, datasets[2])
    # when you want to plot vecs to 2-dimension, you can use below code
    plot_t_sne(t=2, X=hashed_vecs.eval(), y=datasets[2][1].eval())
