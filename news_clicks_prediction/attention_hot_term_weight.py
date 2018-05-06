#!/usr/bin/env python
# -*- coding: utf-8 -*-  

# Copyright (c) 2017 - xiongjiezk <xiongjiezk@163.com>


import tensorflow as tf

from keras.engine import Layer, InputSpec
from keras import regularizers, initializers, constraints
from keras import backend as K

class AttentionHotTermWeight(Layer):

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

    def __init__(self,partition,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        super(AttentionHotTermWeight, self).__init__(**kwargs)

        self.partition = partition
        # self.alpha = 0.9
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias


    def build(self, input_shape):
        assert len(input_shape) == 3

        self.kernel_1 = self.add_weight((self.partition, 1,),
                                 initializer=self.init,
                                 name='{}_W_1'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.kernel_2 = self.add_weight(((input_shape[2] - self.partition), 1,),
                                      initializer=self.init,
                                      name='{}_W_2'.format(self.name),
                                      regularizer=self.W_regularizer,
                                      constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        # word context vector uw
        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionHotTermWeight, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        h = x[:, :, :self.partition]
        t = x[:, :, self.partition:]
        W_w_dot_h_it = K.dot(h, self.kernel_1)
        W_w_dot_h_it = K.squeeze(W_w_dot_h_it, -1)

        W_w_dot_t_it = K.dot(t, self.kernel_2)
        W_w_dot_t_it = K.squeeze(W_w_dot_t_it, -1)

        # W_w_dot_ht_it = tf.multiply(W_w_dot_h_it, W_w_dot_t_it) + self.b  # (batch, 40) + (40,)
        W_w_dot_ht_it = W_w_dot_h_it + W_w_dot_t_it + self.b
        uit = K.tanh(W_w_dot_ht_it)

        uit_dot_uw = uit * self.u
        ait = K.exp(uit_dot_uw)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ait = mask*ait

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number epsilon to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        return ait
        # weighted_input = h * ait
        # # sentence vector si is returned
        # return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1], 1

    def compute_output_shape(self, input_shape):
        """Shape transformation logic so Keras can infer output shape
        """
        return (input_shape[0], input_shape[1], 1,)
