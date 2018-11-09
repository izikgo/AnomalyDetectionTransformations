from keras.models import Model
from keras.layers import Input, Conv2D, Activation, BatchNormalization, Flatten, Dense, Conv2DTranspose, Reshape
from utils import get_channels_axis


def conv_encoder(input_side=32, n_channels=3, representation_dim=256, representation_activation='tanh',
                 intermediate_activation='relu'):
    nf = 64
    input_shape = (n_channels, input_side, input_side) if get_channels_axis() == 1 else (input_side, input_side,
                                                                                         n_channels)

    x_in = Input(shape=input_shape)
    enc = x_in

    # downsample x0.5
    enc = Conv2D(nf, kernel_size=(3, 3), strides=(2, 2), padding='same')(enc)
    enc = BatchNormalization(axis=get_channels_axis())(enc)
    enc = Activation(intermediate_activation)(enc)

    # downsample x0.5
    enc = Conv2D(nf * 2, kernel_size=(3, 3), strides=(2, 2), padding='same')(enc)
    enc = BatchNormalization(axis=get_channels_axis())(enc)
    enc = Activation(intermediate_activation)(enc)

    # downsample x0.5
    enc = Conv2D(nf * 4, kernel_size=(3, 3), strides=(2, 2), padding='same')(enc)
    enc = BatchNormalization(axis=get_channels_axis())(enc)
    enc = Activation(intermediate_activation)(enc)

    if input_side == 64:
        # downsample x0.5
        enc = Conv2D(nf * 8, kernel_size=(3, 3), strides=(2, 2), padding='same')(enc)
        enc = BatchNormalization(axis=get_channels_axis())(enc)
        enc = Activation(intermediate_activation)(enc)

    enc = Flatten()(enc)
    rep = Dense(representation_dim, activation=representation_activation)(enc)

    return Model(x_in, rep)


def conv_decoder(output_side=32, n_channels=3, representation_dim=256, activation='relu'):
    nf = 64

    rep_in = Input(shape=(representation_dim,))

    g = Dense(nf * 4 * 4 * 4)(rep_in)
    g = BatchNormalization(axis=-1)(g)
    g = Activation(activation)(g)

    conv_shape = (nf * 4, 4, 4) if get_channels_axis() == 1 else (4, 4, nf * 4)
    g = Reshape(conv_shape)(g)

    # upsample x2
    g = Conv2DTranspose(nf * 2, kernel_size=(3, 3), strides=(2, 2), padding='same')(g)
    g = BatchNormalization(axis=get_channels_axis())(g)
    g = Activation(activation)(g)

    # upsample x2
    g = Conv2DTranspose(nf, kernel_size=(3, 3), strides=(2, 2), padding='same')(g)
    g = BatchNormalization(axis=get_channels_axis())(g)
    g = Activation(activation)(g)

    if output_side == 64:
        # upsample x2
        g = Conv2DTranspose(nf, kernel_size=(3, 3), strides=(2, 2), padding='same')(g)
        g = BatchNormalization(axis=get_channels_axis())(g)
        g = Activation(activation)(g)

    # upsample x2
    g = Conv2DTranspose(n_channels, kernel_size=(3, 3), strides=(2, 2), padding='same')(g)
    g = Activation('tanh')(g)

    return Model(rep_in, g)


