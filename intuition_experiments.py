import itertools
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from transformations import SimpleTransformer
from utils import load_mnist
from keras.utils import to_categorical
from keras.layers import Flatten, Conv2D, Dense, BatchNormalization, MaxPool2D, Input, Lambda, average
from keras.models import Sequential, Model
import keras.backend as K
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = load_mnist()
# scale to be in [0, 1]
x_train = (x_train + 1) / 2.
x_test = (x_test + 1) / 2.

single_class_ind = 3
anomaly_class_ind = 0

x_train_single = x_train[y_train == single_class_ind]
x_test_single = x_test[y_test == single_class_ind]
x_test_anomaly = x_test[y_test == anomaly_class_ind]

transformer = SimpleTransformer()
transformations_inds = np.tile(np.arange(transformer.n_transforms), len(x_train_single))
x_train_single_transformed = transformer.transform_batch(np.repeat(x_train_single, transformer.n_transforms, axis=0),
                                                         transformations_inds)


mdl = Sequential([Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 1), activation='relu'),
                  BatchNormalization(axis=-1),
                  MaxPool2D(),
                  Flatten(),
                  Dense(10, activation='relu'),
                  BatchNormalization(axis=-1),
                  Dense(transformer.n_transforms, activation='softmax')])

mdl.compile('adam',
            'categorical_crossentropy',
            ['acc'])

batch_size = 64
mdl.fit(x=x_train_single_transformed,
        y=to_categorical(transformations_inds),
        batch_size=batch_size,
        validation_split=0.1,
        epochs=10)

single_class_preds = np.zeros((len(x_test_single), transformer.n_transforms))
for t in range(transformer.n_transforms):
    single_class_preds[:, t] = mdl.predict(transformer.transform_batch(x_test_single, [t] * len(x_test_single)),
                                           batch_size=batch_size)[:, t]
single_class_scores = single_class_preds.mean(axis=-1)

anomaly_class_preds = np.zeros((len(x_test_anomaly), transformer.n_transforms))
for t in range(transformer.n_transforms):
    anomaly_class_preds[:, t] = mdl.predict(transformer.transform_batch(x_test_anomaly, [t] * len(x_test_anomaly)),
                                            batch_size=batch_size)[:, t]
anomaly_class_scores = anomaly_class_preds.mean(axis=-1)


def affine(x, is_flip, k_rotate):
    return tf.image.rot90(tf.image.flip_left_right(x) if is_flip else x,
                          k=k_rotate)


x_in = Input(batch_shape=mdl.input_shape)
transformations_sm_responses = [mdl(Lambda(affine, arguments={'is_flip': is_flip, 'k_rotate': k_rotate})(x_in))
                                for is_flip, k_rotate in itertools.product((False, True), range(4))]
out = average([Lambda(lambda sm_res: sm_res[:, j:j+1])(tens) for j, tens in enumerate(transformations_sm_responses)])


inference_mdl = Model(x_in, out)

grads_tensor = K.gradients([inference_mdl.output], [inference_mdl.input])[0]
grads_fn = K.function([inference_mdl.input], [grads_tensor])


def optimize_anomaly_images():
    for im_ind in range(len(x_test_anomaly)):
        im = x_test_anomaly[im_ind:im_ind+1].copy()

        eta = 5
        for _ in range(200):
            grads = grads_fn([im])[0]
            grads[np.abs(grads * im) < np.percentile(np.abs(grads * im), 80)] = 0
            im_diff = grads * eta
            im_diff *= 0.99
            im += im_diff
            im = gaussian_filter(im, 0.28)

        im = np.clip(im, 0, 1)
        im[im < np.percentile(np.abs(im), 80)] = 0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2))
        ax1.imshow(x_test_anomaly[im_ind].squeeze(), cmap='Greys_r')
        ax1.grid(False)
        ax1.tick_params(which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax2.imshow(im.squeeze(), cmap='Greys_r')
        ax2.grid(False)
        ax2.tick_params(which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        fig.savefig('0_{}.png'.format(im_ind))
        plt.close()
        print('0_3_{} done'.format(im_ind))


def optimize_normal_images():
    for im_ind in range(len(x_train_single)):
        im = x_train_single[im_ind:im_ind+1].copy()

        eta = 5
        for _ in range(200):
            grads = grads_fn([im])[0]
            grads[np.abs(grads * im) < np.percentile(np.abs(grads * im), 80)] = 0
            im_diff = grads * eta
            im_diff *= 0.99
            im += im_diff
            im = gaussian_filter(im, 0.28)

        im = np.clip(im, 0, 1)
        im[im < np.percentile(np.abs(im), 80)] = 0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2))
        ax1.imshow(x_train_single[im_ind].squeeze(), cmap='Greys_r')
        ax1.grid(False)
        ax1.tick_params(which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax2.imshow(im.squeeze(), cmap='Greys_r')
        ax2.grid(False)
        ax2.tick_params(which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        fig.savefig('3_{}.png'.format(im_ind))
        plt.close()
        print('3_3_{} done'.format(im_ind))


optimize_normal_images()
optimize_anomaly_images()
