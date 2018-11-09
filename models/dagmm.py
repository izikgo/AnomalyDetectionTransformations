import numpy as np
from keras.models import Input, Model
from keras.layers import concatenate, Lambda, Layer, add
import keras.backend as K
import tensorflow as tf


class GaussianMixtureComponent(Layer):
    def __init__(self, lambd_diag=0.005, **kwargs):
        self.lambd_diag = lambd_diag
        super().__init__(**kwargs)

    def build(self, input_shapes):
        z_shape, _ = input_shapes
        self.phi = self.add_weight(name='phi',
                                   shape=(1,),
                                   initializer='ones',
                                   trainable=False)
        self.mu = self.add_weight(name='mu',
                                  shape=(1, z_shape[1]),
                                  initializer='uniform',
                                  trainable=False)
        self.sigma = self.add_weight(name='sig',
                                     shape=(z_shape[1], z_shape[1]),
                                     initializer='identity',
                                     trainable=False)
        super().build(input_shapes)

    def call(self, inputs, training=None):
        z, gamma_k = inputs

        gamma_k_sum = K.sum(gamma_k)
        est_phi = K.mean(gamma_k, axis=0)
        est_mu = K.dot(K.transpose(gamma_k), z) / gamma_k_sum
        est_sigma = K.dot(K.transpose(z - est_mu),
                          gamma_k * (z - est_mu)) / gamma_k_sum

        est_sigma = est_sigma + (K.random_normal(shape=(K.int_shape(z)[1], 1), mean=1e-3, stddev=1e-4) * K.eye(K.int_shape(z)[1]))

        self.add_update(K.update(self.phi, est_phi), inputs)
        self.add_update(K.update(self.mu, est_mu), inputs)
        self.add_update(K.update(self.sigma, est_sigma), inputs)

        est_sigma_diag_inv = K.eye(K.int_shape(self.sigma)[0]) / est_sigma
        self.add_loss(self.lambd_diag * K.sum(est_sigma_diag_inv), inputs)

        phi = K.in_train_phase(est_phi, self.phi, training)
        mu = K.in_train_phase(est_mu, self.mu, training)
        sigma = K.in_train_phase(est_sigma, self.sigma, training)
        return GaussianMixtureComponent._calc_component_density(z, phi, mu, sigma)

    @staticmethod
    def _calc_component_density(z, phi, mu, sigma):
        sig_inv = tf.matrix_inverse(sigma)
        sig_sqrt_det = K.sqrt(tf.matrix_determinant(2 * np.pi * sigma) + K.epsilon())
        density = phi * (K.exp(-0.5 * K.sum(K.dot(z - mu, sig_inv) * (z - mu),
                                            axis=-1,
                                            keepdims=True)) / sig_sqrt_det) + K.epsilon()

        return density

    def compute_output_shape(self, input_shapes):
        z_shape, gamma_k_shape = input_shapes
        return z_shape[0], 1


def create_dagmm_model(encoder, decoder, estimation_encoder, lambd_diag=0.005):
    x_in = Input(batch_shape=encoder.input_shape)
    zc = encoder(x_in)

    decoder.name = 'reconstruction'
    x_rec = decoder(zc)
    euclid_dist = Lambda(lambda args: K.sqrt(K.sum(K.batch_flatten(K.square(args[0] - args[1])),
                                                   axis=-1, keepdims=True) /
                                             K.sum(K.batch_flatten(K.square(args[0])),
                                                   axis=-1, keepdims=True)),
                         output_shape=(1,))([x_in, x_rec])
    cos_sim = Lambda(lambda args: K.batch_dot(K.l2_normalize(K.batch_flatten(args[0]), axis=-1),
                                              K.l2_normalize(K.batch_flatten(args[1]), axis=-1),
                                              axes=-1),
                     output_shape=(1,))([x_in, x_rec])

    zr = concatenate([euclid_dist, cos_sim])
    z = concatenate([zc, zr])

    gamma = estimation_encoder(z)

    gamma_ks = [Lambda(lambda g: g[:, k:k + 1], output_shape=(1,))(gamma)
                for k in range(estimation_encoder.output_shape[-1])]

    components = [GaussianMixtureComponent(lambd_diag)([z, gamma_k])
                  for gamma_k in gamma_ks]
    density = add(components) if len(components) > 1 else components[0]
    energy = Lambda(lambda dens: -K.log(dens), name='energy')(density)

    dagmm = Model(x_in, [x_rec, energy])

    return dagmm
