import numpy as np
from keras.optimizers import Adam
from keras.models import Model
from keras.layers.merge import subtract
from keras.utils.generic_utils import Progbar
from keras.engine.topology import Input, Layer
from keras.callbacks import CallbackList
import keras.backend as K


class GradPenLayer(Layer):
    def call(self, inputs, **kwargs):
        interp_in, critic_interp_score_in = inputs[0], inputs[1]
        interp_critic_grad = K.batch_flatten(K.gradients(critic_interp_score_in, [interp_in])[0])
        interp_critic_grad_norm = K.sqrt(K.sum(K.square(interp_critic_grad), axis=-1, keepdims=True))
        return K.square(interp_critic_grad_norm - 1.)  # two sided regularisation
        # return K.square(K.relu(interp_critic_grad_norm - 1.))  # one sided regularisation

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1


def train_wgan_with_grad_penalty(prior_gen, generator, data_gen, critic, batch_size, epochs,
                                 batches_per_epoch=100, optimizer=Adam(lr=1e-4, beta_1=0, beta_2=0.9),
                                 grad_pen_coef=10., critic_gen_train_ratio=2, callbacks=None):
    # build model to train the critic
    data_shape = critic.input_shape[1:]
    real_critic_input = Input(shape=data_shape, name='real_in')
    fake_critic_input = Input(shape=data_shape, name='fake_in')
    interp_critic_input = Input(shape=data_shape, name='interp_in')

    real_critic_score = critic(real_critic_input)
    fake_critic_score = critic(fake_critic_input)
    interp_critic_score = critic(interp_critic_input)

    critic_loss = subtract([fake_critic_score, real_critic_score])
    gradient_penalty = GradPenLayer()([interp_critic_input, interp_critic_score])

    critic_train_mdl = Model([real_critic_input, fake_critic_input, interp_critic_input],
                             [critic_loss, gradient_penalty])

    critic_train_mdl.compile(optimizer=optimizer,
                             loss=lambda y_true, y_pred: y_pred,
                             loss_weights=[1., grad_pen_coef])

    # build model to train generator
    prior_input = Input(shape=generator.input_shape[1:], name='prior_in')
    critic.trainable = False
    critic_on_generator_score = critic(generator(prior_input))
    generator_train_mdl = Model(prior_input, critic_on_generator_score)
    generator_train_mdl.compile(optimizer=optimizer, loss=lambda y_true, y_pred: -y_pred)

    # init callbacks
    callbacks = callbacks or []
    callbacks = CallbackList(callbacks)
    callbacks.set_model({'generator': generator, 'critic': critic})
    callbacks.set_params({
        'batch_size': batch_size,
        'epochs': epochs,
        'steps': batches_per_epoch,
        'samples': batches_per_epoch * batch_size,
        'prior_gen': prior_gen,
        'data_gen': data_gen,
    })

    # train
    print('Training on {} samples for {} epochs'.format(batches_per_epoch * batch_size, epochs))
    callbacks.on_train_begin()
    for e in range(epochs):
        print('Epoch {}/{}'.format(e + 1, epochs))
        callbacks.on_epoch_begin(e)
        progbar = Progbar(target=batches_per_epoch*batch_size)
        dummy_y = np.array([None]*batch_size)
        for b in range(batches_per_epoch):
            callbacks.on_batch_begin(b)
            batch_losses = np.zeros(shape=3)
            for critic_upd in range(critic_gen_train_ratio):
                real_batch = data_gen(batch_size)
                fake_batch = generator.predict(prior_gen(batch_size))
                weights = np.random.uniform(size=batch_size)
                weights = weights.reshape((-1,) + (1,)*(len(real_batch.shape)-1))
                interp_batch = weights * real_batch + (1. - weights) * fake_batch

                x_batch = {'real_in': real_batch, 'fake_in': fake_batch, 'interp_in': interp_batch}
                cur_losses = np.array(critic_train_mdl.train_on_batch(x=x_batch, y=[dummy_y, dummy_y]))
                batch_losses += cur_losses

            generator_train_mdl.train_on_batch(x=prior_gen(batch_size), y=dummy_y)

            losses_names = ('total_loss', 'critic_loss', 'gradient_pen')
            progbar.add(batch_size, zip(losses_names, batch_losses))
            callbacks.on_batch_end(b)

        progbar.update(batches_per_epoch*batch_size)
        callbacks.on_epoch_end(e)

    callbacks.on_train_end()


def scores_from_adgan_generator(x_test, prior_gen, generator, n_seeds=8, k=5, z_lr=0.25, gen_lr=5e-5):
    generator.trainable = True
    initial_weights = generator.get_weights()

    gen_opt = Adam(lr=gen_lr, beta_1=0.5)
    z_opt = Adam(lr=z_lr, beta_1=0.5)

    x_ph = K.placeholder((1,)+x_test.shape[1:])
    z = K.variable(prior_gen(1))
    rec_loss = K.mean(K.square(x_ph - generator(z)))
    z_train_fn = K.function([x_ph], [rec_loss], updates=z_opt.get_updates(rec_loss, [z]))
    g_train_fn = K.function([x_ph, K.learning_phase()], [rec_loss],
                            updates=gen_opt.get_updates(rec_loss, generator.trainable_weights))

    gen_opt_initial_params = gen_opt.get_weights()
    z_opt_initial_params = z_opt.get_weights()

    scores = []
    for x in x_test:
        x = np.expand_dims(x, axis=0)
        losses = []
        for j in range(n_seeds):
            K.set_value(z, prior_gen(1))
            generator.set_weights(initial_weights)
            gen_opt.set_weights(gen_opt_initial_params)
            z_opt.set_weights(z_opt_initial_params)
            for _ in range(k):
                z_train_fn([x])
                g_train_fn([x, 1])
            loss = z_train_fn([x])[0]
            losses.append(loss)

        score = -np.mean(losses)
        scores.append(score)

    return np.array(scores)
