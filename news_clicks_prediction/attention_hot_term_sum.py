#!/usr/bin/env python
# -*- coding: utf-8 -*-  

# Copyright (c) 2017 - xiongjiezk <xiongjiezk@163.com>


import tensorflow as tf

from keras.engine import Layer, InputSpec
from keras import regularizers, initializers, constraints
from keras import backend as K

class AttentionHotTermSum(Layer):
    """
    clac final representation from intermediate attention weight value,
    """


    def __init__(self,partition,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        super(AttentionHotTermSum, self).__init__(**kwargs)

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


    def build(self, input_shape):
        super(AttentionHotTermSum, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        xx = x[:, :, :self.partition]
        ait = x[:, :, self.partition:]
        weighted_input = xx * ait
        # sentence vector si is returned
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.partition

    def compute_output_shape(self, input_shape):
        """Shape transformation logic so Keras can infer output shape
        """
        return (input_shape[0], self.partition,)
