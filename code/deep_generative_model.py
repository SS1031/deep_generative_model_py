#!/usr/bin/env python
# -*- encoding: utf-8 -*-


'''
    ____
   / __ \___  ___  ____
  / / / / _ \/ _ \/ __ \
 / /_/ /  __/  __/ /_/ /
/_____/\___/\___/ .___/\
               /_/
          ______                            __  _
         / ____/__  ____  ____ __________ _/ /_(_)   _____
        / / __/ _ \/ __ \/ __ `/ ___/ __ `/ __/ / | / / _ \
       / /_/ /  __/ / / / /_/ / /  / /_/ / /_/ /| |/ /  __/
      /_____/\___/_/ /_/\__,_/_/   \__,_/\__/_/ |___/\___/
                                        __  ___          __     __
                                       /  |/  /___  ____/ /__  / /
                                      / /|_/ / __ \/ __  / _ \/ /
                                     / /  / / /_/ / /_/ /  __/ /
                                    /_/  /_/\____/\__,_/\___/_/

Deep Generative Model
Reference : http://www.cs.toronto.edu/~rsalakhu/papers/topics.pdf
'''

import os
import sys
import time
import PIL.Image

import numpy
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams
from rbm import RBM

from utils import load_data
from utils import tile_raster_images


class DGM(object):
    """
    Deep Generative Model Class
    """
    def __init__(self, numpy_rng, theano_rng=None, input=None,
                 input_size=784, hidden_layers_sizes=[500, 500],
                 W_list=None, hbias_list=None, vbias_list=None
                 ):
        """
        @type   numpy_rng   numpy.random.RandomState
        @param  numpy_rng   random generator for initialize weight matrix
        @type   theano_rng  theano.tensor.shared_randomstreams.RandomStreams
        @param  theano_rng  theano random generator
        @type   input       theano.tensor.TensorType
        @param  input       symbolic variables of input
        @type   input_size  int
        @param  input_size  size of input
        @type   hidden_layers_sizes     list
        @param  hidden_layers_sizes     sizes of hidden layers
        @type   W_list      list
        @param  W_list      weight matrix list, auto initialize if None
        @type   hbias_list  list
        @param  hbias_list  hidden layer bias list
        @type   vbias_list   list
        @param  vbias_list   visible layer bias list
        """

        if input is None:
            self.x = T.matrix('x')
        else:
            self.x = input

        self.y = T.ivector('y')

        # set number of layer
        self.n_layers = len(hidden_layers_sizes)

        self.hbias_list = []
        self.vbias_list = []
        self.params = []
        self.rbm_layers = []

        if not theano_rng:
            # generate random variables
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

        # if W_list is none, initialize weight matrix
        if not W_list:
            # initialize W list
            self.W_list = []
            self.W_T_list = []

            for i in xrange(self.n_layers):
                # initializing for layer length

                # use below layer output except first layer,
                # if it's on the first layer use inputs
                if i != 0:
                    layer_input = self.rbm_layers[-1].output
                    n_input = hidden_layers_sizes[i - 1]
                else:
                    layer_input = self.x
                    n_input = input_size

                # each layer is initialized by RBM for pretraining by RBM
                rbm_layer = RBM(numpy_rng=numpy_rng,
                                theano_rng=theano_rng,
                                input=layer_input,
                                n_visible=n_input,
                                n_hidden=hidden_layers_sizes[i],
                                W=None,
                                hbias=None)

                # save each layers
                self.rbm_layers.append(rbm_layer)

                # add each layers weight and transposed weight for finetuning
                self.W_list.append(rbm_layer.W)
                self.W_T_list.append(rbm_layer.W.T)

                # add params to params list for finetuning
                self.params.append(rbm_layer.W)
                self.hbias_list.append(rbm_layer.hbias)
                self.params.append(rbm_layer.hbias)
                self.vbias_list.append(rbm_layer.vbias)
                self.params.append(rbm_layer.vbias)

    def get_pretrain_fns(self, train_set_x, batch_size, k=1):
        '''
        create pretrainig functions list

        @type   train_set_x  theano.tensor.TensorType
        @param  train_set_x  training dataset for rbm
        @type   batch_size   int
        @param  batch_size   batch size of batch trainig
        @type   theano_rng   theano.tensor.shared_randomstreams.RandomStreams
        @param  theano_rng   random variables

        @return pretraining functions list
        '''

        # index of minibatch
        index = T.lscalar('index')
        learning_rate = T.scalar('lr')

        # start index of batches
        batch_begin = index * batch_size
        # end index of batches
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        # create pretraining functions for each layers
        for rbm in self.rbm_layers:
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=k)

            # make theano function
            fn = theano.function(inputs=[index,
                                 theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x:
                                         train_set_x[batch_begin:batch_end]})

            pretrain_fns.append(fn)

        return pretrain_fns

    def encode(self, input):
        """
        encode input, consider whole network as autoencoder

        @type   input   theano.tensor.TensorType
        @param  input   network input values
        @return         encoded input data, 128 dimension
        """

        output =\
            T.nnet.sigmoid(T.dot(input, self.W_list[0]) + self.hbias_list[0])

        for i in range(self.n_layers)[1:]:
            output =\
                T.nnet.sigmoid(
                    T.dot(output, self.W_list[i]) + self.hbias_list[i]
                    )

        return output

    def decode(self, encoded_input):
        """
        decode encoded input data, consider the whole network as autoencoder

        @type   encoded_input    theano.tensor.TensorType
        @param  encoded_input    encoded input data

        @return                  decoded data, dimension is same as
                                 input data dimension
        """
        output =\
            T.nnet.sigmoid(
                T.dot(encoded_input, self.W_T_list[-1]) + self.vbias_list[-1]
                )

        for i in range(self.n_layers)[-2::-1]:
            output =\
                T.nnet.sigmoid(
                    T.dot(output, self.W_T_list[i]) + self.vbias_list[i]
                )

        return output

    def get_autoencoder_cost(self):
        """
        create cost function of the whole network autoencode

        @return cost of the whole network autoencoder
        """
        z = self.decode(self.encode(self.x))
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        return T.mean(L)

    def get_cost_updates(self, learning_rate):
        """
        create cost and parameters updates
        this cost and updates will be used at finetuning

        @type   learning_rate   float
        @param  learning_rate   learning rate of finetuning

        @return
            cost                cost functions (cross entropy)
            updates             parameters update functions
        """
        # get cost
        cost = self.get_autoencoder_cost()
        # gradients of cost by params
        gparams = T.grad(cost, self.params)
        # update functions of parameters
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        """
        generate a finetuning functions.
        'train_fn' is for finetuning training
        'valid_score' is for validation steps

        @type   datasets    list of pairs of theano.tensor.TensorType
        @param  datasets    'train', 'valid', 'test', datasets
        @type   batch_size  int
        @param  batch_size  size of a minibatch
        @type   learning_rate   float
        @param  learning_rate   learning rate used during finetune stage

        @return
            train_fns       theano function for finetuning train
            valid_score     validation score of all validation dataset
        """
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch
        cost, updates = self.get_cost_updates(learning_rate)

        '''
        gparams = T.grad(cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))
        '''

        # create train function
        train_fn =\
            theano.function(inputs=[index],
                            outputs=cost,
                            updates=updates,
                            givens={self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size]}
                            )

        # create valid score function
        valid_score_i =\
            theano.function(inputs=[index],
                            outputs=cost,
                            givens={self.x: valid_set_x[index * batch_size:
                                    (index + 1) * batch_size]}
                            )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        return train_fn, valid_score


def test_DGM(pretrain_lr=0.1, pretraining_epochs=10,
             finetune_lr=0.1, finetuning_epochs=1,
             k=1, datasets=None,
             batch_size=20, output_folder='../result'):
    '''
    testing Deep Generative Model
    use hand written data(MNIST), pretraing and finetuning with
    MNIST data(784dimension).
    after finetuning, try reconstruction of input data using the whole
    network autoencoder.

    @type   pretrain_lr     float
    @param  pretrain_lr     learning rate for pretraining
    @type   pretraining_epochs  int
    @param  pretraining_epochs  pretrain learning epochs
    @type   finetuning_epochs   int
    @param  finetuning_epochs   finetune learning epochs
    @type   k               int
    @param  k               for CD-k
    @type   dataset         list of theano.tensor.TensorType
    @param  dataset         'train', 'valid', 'test' datasets
    @type   batch_size      int
    @param  batch_size      size of batch traning
    @type   output_folder   string
    @param  output_folder
    '''

    # datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # set ouput folder
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # calc number of minibatches
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # define input x
    x = T.matrix('x')

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize dgm
    dgm = DGM(numpy_rng=rng, theano_rng=theano_rng,
              input=x, input_size=28 * 28,
              hidden_layers_sizes=[500, 500, 128])

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = dgm.get_pretrain_fns(train_set_x=train_set_x,
                                           batch_size=batch_size,
                                           k=k)

    print '... pre-training the model'
    start_time = time.clock()
    # Pre-train layer-wise
    for i in xrange(dgm.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                # add all cost over all train dataset
                c.append(pretraining_fns[i](index=batch_index, lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            # average cost of all train batches
            print numpy.mean(c)

    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################
    print '... building finetuning model'
    train_fn, validate_model =\
        dgm.build_finetune_functions(datasets=datasets,
                                     batch_size=batch_size,
                                     learning_rate=finetune_lr)

    # TODO: use validation frequency proper
    patience = 4 * n_train_batches
    validation_frequency = min(n_train_batches, patience / 2)
    best_validation_loss = numpy.inf

    start_time = time.clock()

    print '... start finetuning'
    # start epochs
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
                       n_train_batches,
                       this_validation_loss))

                if this_validation_loss < best_validation_loss:
                    # save the best validation costs and params
                    best_validation_loss = this_validation_loss
                    # TODO: use the best_params
                    # best_params = dgm.params
        print 'Finetuning epoch %d, cost ' %\
              epoch, numpy.mean(minibatch_avg_cost)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print training_time

    #############################
    # test image reconstruction #
    #############################
    # save original image
    image = PIL.Image.fromarray(tile_raster_images(X=test_set_x[0:100].eval(),
                                                   img_shape=(28, 28),
                                                   tile_shape=(10, 10),
                                                   tile_spacing=(1, 1)
                                                   )
                                )
    img_f_name = 'DGM_origin.png'
    image.save(img_f_name)

    # input the original image data, then reconstruct
    y = dgm.encode(test_set_x[0:100])
    z = dgm.decode(y)

    image = PIL.Image.fromarray(tile_raster_images(X=z.eval(),
                                                   img_shape=(28, 28),
                                                   tile_shape=(10, 10),
                                                   tile_spacing=(1, 1)
                                                   )
                                )

    img_f_name = 'DGM_reconstruction_img_preEpoch_%d_fineEpoch_%d.png' %\
                 (pretraining_epochs, finetuning_epochs)
    image.save(img_f_name)


if __name__ == '__main__':
    file_name = 'mnist.pkl.gz'
    dataset = load_data(dataset=file_name)
    test_DGM(pretraining_epochs=1, finetuning_epochs=1, datasets=dataset)
