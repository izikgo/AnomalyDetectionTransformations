from keras.layers import *
from keras.models import Model
import keras.backend as K


class Prior(Layer):
    def __init__(self,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Prior, self).__init__(**kwargs)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.bias = None

    def build(self, input_shape):
        self.bias = self.add_weight(shape=input_shape[1:],
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)
        super(Prior, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1

    def call(self, x, **kwargs):
        return K.sum(K.batch_flatten(K.square(K.bias_add(x, -self.bias))), axis=-1, keepdims=True)


def create_energy_model(encoder_mdl):
    x_in = Input(batch_shape=encoder_mdl.input_shape)

    encoded = encoder_mdl(x_in)
    prior = Prior()(x_in)
    energy = Lambda(lambda args: 0.5*args[0] - K.sum(K.batch_flatten(args[1]), axis=-1, keepdims=True),
                    output_shape=(1,))([prior, encoded])
    return Model(x_in, energy)


def create_reconstruction_model(energy_mdl):
    x_in = Input(batch_shape=energy_mdl.input_shape)
    x = GaussianNoise(stddev=.5)(x_in)  # only active in training
    energy = energy_mdl(x)
    rec = Lambda(lambda args: args[1] - K.gradients(args[0], args[1]),
                 output_shape=energy_mdl.input_shape[1:])([energy, x])

    return Model(x_in, rec)
