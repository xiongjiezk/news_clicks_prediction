#!/usr/bin/env python
# -*- coding: utf-8 -*-  


import tensorflow as tf

from keras.engine import Layer, InputSpec
from keras import regularizers, initializers, constraints
from keras import backend as K

class AttentionSum(Layer):

    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.

    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]

    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    :param kwargs:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.

    refer https://github.com/fchollet/keras/issues/4962
    refer https://gist.github.com/rmdort/596e75e864295365798836d9e8636033

    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
    """

    def __init__(self, partition,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.partition = partition
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionSum, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        # self.kernel = self.add_weight((input_shape[2], 1,),
        #                          initializer=self.init,
        #                          name='{}_W'.format(self.name),
        #                          regularizer=self.W_regularizer,
        #                          constraint=self.W_constraint)
        # if self.bias:
        #     self.b = self.add_weight((input_shape[1],),
        #                              initializer='zero',
        #                              name='{}_b'.format(self.name),
        #                              regularizer=self.b_regularizer,
        #                              constraint=self.b_constraint)
        #
        # # word context vector uw
        # self.u = self.add_weight((input_shape[1],),
        #                          initializer=self.init,
        #                          name='{}_u'.format(self.name),
        #                          regularizer=self.u_regularizer,
        #                          constraint=self.u_constraint)

        super(AttentionSum, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):  # x word vector, shape: (?, 80,100)
        # u * tanh(wx+b)
        # W_w_dot_h_it = K.dot(x, self.kernel)  # (?, 80, 100) . (100, 1) --> (?, 80, 1)
        # W_w_dot_h_it = K.squeeze(W_w_dot_h_it, -1)  # (?, 80, 1) --> (?, 80)
        # W_w_dot_h_it = W_w_dot_h_it + self.b  # (?, 80) + (80, ) --> (?, 80)
        # uit = K.tanh(W_w_dot_h_it)  # (?, 80)
        # uit_dot_uw = uit * self.u  # (?, 80) * (80, ) --> (?, 80)
        # ait = K.exp(uit_dot_uw)  # (?, 80)
        #
        # if mask is not None:
        #     mask = K.cast(mask, K.floatx())
        #     ait = mask*ait  # (?, 80)
        #
        # # in some cases especially in the early stages of training the sum may be almost zero
        # # and this results in NaN's. A workaround is to add a very small positive number epsilon to the sum.
        # # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        # ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())  # (?, 80)
        # ait = K.expand_dims(ait)  # (?, 80, 1)
        xx = x[:, :, :self.partition]
        ait = x[:, :, self.partition:]
        weighted_input = x * ait  # (?, 80, 100) * (?, 80, 1) --> (?, 80, 100)
        # sentence vector si is returned
        return K.sum(weighted_input, axis=1)  # (?, 100)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.partition

    def compute_output_shape(self, input_shape):
        """Shape transformation logic so Keras can infer output shape
        """
        return (input_shape[1], self.partition, )

