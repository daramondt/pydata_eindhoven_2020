import tensorflow as tf
from tensorflow.keras import activations, initializers, constraints, regularizers
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import scipy.sparse as sp
import numpy as np


class GraphConvolution(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0].as_list()
        output_shape = (features_shape[0], features_shape[1], self.units)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        features_shape = input_shapes[0].as_list()
        input_dim = features_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim,
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        features = inputs[0]
        adjacency = inputs[1]

        convolution = K.batch_dot(K.permute_dimensions(adjacency, (0,2,1)), features)
        output = K.dot(convolution, self.kernel)

        if self.use_bias:
            output += self.bias
        return self.activation(output)

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PersonalizedPageRank(Layer):
    """Basic PPR layer as in https://arxiv.org/pdf/1810.05997.pdf"""
    def __init__(self, alpha, niter, keep_prob, activation=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(PersonalizedPageRank, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.supports_masking = True

        self.alpha = alpha
        self.niter = niter
        self.keep_prob = keep_prob

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0].as_list()
        output_shape = (features_shape[0], self.units)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        self.built = True

    def call(self, inputs, mask=None):
        adj_matrix = inputs[1]

        A_hat = K.permute_dimensions((1 - self.alpha) * adj_matrix, (0,2,1))

        Z = inputs[0]

        Zs_prop = Z
        for _ in range(self.niter):
            A_drop = tf.nn.dropout(A_hat, self.keep_prob)
            Zs_prop = K.batch_dot(A_drop, Zs_prop) + self.alpha * Z
        return Zs_prop

    def get_config(self):
        config = {'alpha': self.alpha,
                  'niter': self.niter,
                  'keep_prob': self.keep_prob,
                  'activation': activations.serialize(self.activation)
        }

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
