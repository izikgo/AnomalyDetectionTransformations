from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
from keras.initializers import _compute_fans
from keras.optimizers import SGD
from keras import backend as K


WEIGHT_DECAY = 0.5 * 0.0005


class SGDTorch(SGD):
    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m + g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p - lr * (self.momentum * v + g)
            else:
                new_p = p - lr * v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates


def _get_channels_axis():
    return -1 if K.image_data_format() == 'channels_last' else 1


def _conv_kernel_initializer(shape, dtype=None):
    fan_in, fan_out = _compute_fans(shape)
    stddev = np.sqrt(2. / fan_in)
    return K.random_normal(shape, 0., stddev, dtype)


def _dense_kernel_initializer(shape, dtype=None):
    fan_in, fan_out = _compute_fans(shape)
    stddev = 1. / np.sqrt(fan_in)
    return K.random_uniform(shape, -stddev, stddev, dtype)


def batch_norm():
    return BatchNormalization(axis=_get_channels_axis(), momentum=0.9, epsilon=1e-5,
                              beta_regularizer=l2(WEIGHT_DECAY), gamma_regularizer=l2(WEIGHT_DECAY))


def conv2d(output_channels, kernel_size, strides=1):
    return Convolution2D(output_channels, kernel_size, strides=strides, padding='same', use_bias=False,
                         kernel_initializer=_conv_kernel_initializer, kernel_regularizer=l2(WEIGHT_DECAY))


def dense(output_units):
    return Dense(output_units, kernel_initializer=_dense_kernel_initializer, kernel_regularizer=l2(WEIGHT_DECAY),
                 bias_regularizer=l2(WEIGHT_DECAY))


def _add_basic_block(x_in, out_channels, strides, dropout_rate=0.0):
    is_channels_equal = K.int_shape(x_in)[_get_channels_axis()] == out_channels

    bn1 = batch_norm()(x_in)
    bn1 = Activation('relu')(bn1)
    out = conv2d(out_channels, 3, strides)(bn1)
    out = batch_norm()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout_rate)(out)
    out = conv2d(out_channels, 3, 1)(out)
    shortcut = x_in if is_channels_equal else conv2d(out_channels, 1, strides)(bn1)
    out = add([out, shortcut])
    return out


def _add_conv_group(x_in, out_channels, n, strides, dropout_rate=0.0):
    out = _add_basic_block(x_in, out_channels, strides, dropout_rate)
    for _ in range(1, n):
        out = _add_basic_block(out, out_channels, 1, dropout_rate)
    return out


def create_wide_residual_network(input_shape, num_classes, depth, widen_factor=1, dropout_rate=0.0,
                                 final_activation='softmax'):
    n_channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
    assert ((depth - 4) % 6 == 0), 'depth should be 6n+4'
    n = (depth - 4) // 6

    inp = Input(shape=input_shape)
    conv1 = conv2d(n_channels[0], 3)(inp)  # one conv at the beginning (spatial size: 32x32)
    conv2 = _add_conv_group(conv1, n_channels[1], n, 1, dropout_rate)  # Stage 1 (spatial size: 32x32)
    conv3 = _add_conv_group(conv2, n_channels[2], n, 2, dropout_rate)  # Stage 2 (spatial size: 16x16)
    conv4 = _add_conv_group(conv3, n_channels[3], n, 2, dropout_rate)  # Stage 3 (spatial size: 8x8)

    out = batch_norm()(conv4)
    out = Activation('relu')(out)
    out = AveragePooling2D(8)(out)
    out = Flatten()(out)

    out = dense(num_classes)(out)
    out = Activation(final_activation)(out)

    return Model(inp, out)
