import os
import csv
from collections import defaultdict
from glob import glob
from datetime import datetime
from multiprocessing import Manager, freeze_support, Process
import numpy as np
import scipy.stats
from scipy.special import psi, polygamma
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.model_selection import ParameterGrid
from sklearn.externals.joblib import Parallel, delayed
from keras.models import Model, Input, Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from utils import load_cifar10, load_cats_vs_dogs, load_fashion_mnist, load_cifar100
from utils import save_roc_pr_curve_data, get_class_name_from_index, get_channels_axis
from transformations import Transformer
from models.wide_residual_network import create_wide_residual_network
from models.encoders_decoders import conv_encoder, conv_decoder
from models import dsebm, dagmm, adgan
import keras.backend as K

RESULTS_DIR = ''


def _transformations_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q):
    gpu_to_use = gpu_q.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use

    (x_train, y_train), (x_test, y_test) = dataset_load_fn()

    if dataset_name in ['cats-vs-dogs']:
        transformer = Transformer(16, 16)
        n, k = (16, 8)
    else:
        transformer = Transformer(8, 8)
        n, k = (10, 4)
    mdl = create_wide_residual_network(x_train.shape[1:], transformer.n_transforms, n, k)
    mdl.compile('adam',
                'categorical_crossentropy',
                ['acc'])

    x_train_task = x_train[y_train.flatten() == single_class_ind]
    transformations_inds = np.tile(np.arange(transformer.n_transforms), len(x_train_task))
    x_train_task_transformed = transformer.transform_batch(np.repeat(x_train_task, transformer.n_transforms, axis=0),
                                                           transformations_inds)
    batch_size = 128

    mdl.fit(x=x_train_task_transformed, y=to_categorical(transformations_inds),
            batch_size=batch_size, epochs=int(np.ceil(200/transformer.n_transforms))
            )

    #################################################################################################
    # simplified normality score
    #################################################################################################
    # preds = np.zeros((len(x_test), transformer.n_transforms))
    # for t in range(transformer.n_transforms):
    #     preds[:, t] = mdl.predict(transformer.transform_batch(x_test, [t] * len(x_test)),
    #                               batch_size=batch_size)[:, t]
    #
    # labels = y_test.flatten() == single_class_ind
    # scores = preds.mean(axis=-1)
    #################################################################################################

    def calc_approx_alpha_sum(observations):
        N = len(observations)
        f = np.mean(observations, axis=0)

        return (N * (len(f) - 1) * (-psi(1))) / (
                N * np.sum(f * np.log(f)) - np.sum(f * np.sum(np.log(observations), axis=0)))

    def inv_psi(y, iters=5):
        # initial estimate
        cond = y >= -2.22
        x = cond * (np.exp(y) + 0.5) + (1 - cond) * -1 / (y - psi(1))

        for _ in range(iters):
            x = x - (psi(x) - y) / polygamma(1, x)
        return x

    def fixed_point_dirichlet_mle(alpha_init, log_p_hat, max_iter=1000):
        alpha_new = alpha_old = alpha_init
        for _ in range(max_iter):
            alpha_new = inv_psi(psi(np.sum(alpha_old)) + log_p_hat)
            if np.sqrt(np.sum((alpha_old - alpha_new) ** 2)) < 1e-9:
                break
            alpha_old = alpha_new
        return alpha_new

    def dirichlet_normality_score(alpha, p):
        return np.sum((alpha - 1) * np.log(p), axis=-1)

    scores = np.zeros((len(x_test),))
    observed_data = x_train_task
    for t_ind in range(transformer.n_transforms):
        observed_dirichlet = mdl.predict(transformer.transform_batch(observed_data, [t_ind] * len(observed_data)),
                                         batch_size=1024)
        log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)

        alpha_sum_approx = calc_approx_alpha_sum(observed_dirichlet)
        alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx

        mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)

        x_test_p = mdl.predict(transformer.transform_batch(x_test, [t_ind] * len(x_test)),
                               batch_size=1024)
        scores += dirichlet_normality_score(mle_alpha_t, x_test_p)

    scores /= transformer.n_transforms
    labels = y_test.flatten() == single_class_ind

    res_file_name = '{}_transformations_{}_{}.npz'.format(dataset_name,
                                                 get_class_name_from_index(single_class_ind, dataset_name),
                                                 datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
    save_roc_pr_curve_data(scores, labels, res_file_path)

    mdl_weights_name = '{}_transformations_{}_{}_weights.h5'.format(dataset_name,
                                                           get_class_name_from_index(single_class_ind, dataset_name),
                                                           datetime.now().strftime('%Y-%m-%d-%H%M'))
    mdl_weights_path = os.path.join(RESULTS_DIR, dataset_name, mdl_weights_name)
    mdl.save_weights(mdl_weights_path)

    gpu_q.put(gpu_to_use)


def _train_ocsvm_and_score(params, xtrain, test_labels, xtest):
    return roc_auc_score(test_labels, OneClassSVM(**params).fit(xtrain).decision_function(xtest))


def _raw_ocsvm_experiment(dataset_load_fn, dataset_name, single_class_ind):
    (x_train, y_train), (x_test, y_test) = dataset_load_fn()

    x_train = x_train.reshape((len(x_train), -1))
    x_test = x_test.reshape((len(x_test), -1))

    x_train_task = x_train[y_train.flatten() == single_class_ind]
    if dataset_name in ['cats-vs-dogs']:  # OC-SVM is quadratic on the number of examples, so subsample training set
        subsample_inds = np.random.choice(len(x_train_task), 5000, replace=False)
        x_train_task = x_train_task[subsample_inds]

    pg = ParameterGrid({'nu': np.linspace(0.1, 0.9, num=9),
                        'gamma': np.logspace(-7, 2, num=10, base=2)})

    results = Parallel(n_jobs=6)(
        delayed(_train_ocsvm_and_score)(d, x_train_task, y_test.flatten() == single_class_ind, x_test)
        for d in pg)

    best_params, best_auc_score = max(zip(pg, results), key=lambda t: t[-1])
    best_ocsvm = OneClassSVM(**best_params).fit(x_train_task)
    scores = best_ocsvm.decision_function(x_test)
    labels = y_test.flatten() == single_class_ind

    res_file_name = '{}_raw-oc-svm_{}_{}.npz'.format(dataset_name,
                                                     get_class_name_from_index(single_class_ind, dataset_name),
                                                     datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
    save_roc_pr_curve_data(scores, labels, res_file_path)


def _cae_ocsvm_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q):
    gpu_to_use = gpu_q.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use

    (x_train, y_train), (x_test, y_test) = dataset_load_fn()

    n_channels = x_train.shape[get_channels_axis()]
    input_side = x_train.shape[2]  # channel side will always be at shape[2]
    enc = conv_encoder(input_side, n_channels)
    dec = conv_decoder(input_side, n_channels)
    x_in = Input(shape=x_train.shape[1:])
    x_rec = dec(enc(x_in))
    cae = Model(x_in, x_rec)
    cae.compile('adam', 'mse')

    x_train_task = x_train[y_train.flatten() == single_class_ind]
    x_test_task = x_test[y_test.flatten() == single_class_ind]  # This is just for visual monitoring
    cae.fit(x=x_train_task, y=x_train_task, batch_size=128, epochs=200, validation_data=(x_test_task, x_test_task))

    x_train_task_rep = enc.predict(x_train_task, batch_size=128)
    if dataset_name in ['cats-vs-dogs']:  # OC-SVM is quadratic on the number of examples, so subsample training set
        subsample_inds = np.random.choice(len(x_train_task_rep), 2500, replace=False)
        x_train_task_rep = x_train_task_rep[subsample_inds]

    x_test_rep = enc.predict(x_test, batch_size=128)
    pg = ParameterGrid({'nu': np.linspace(0.1, 0.9, num=9),
                        'gamma': np.logspace(-7, 2, num=10, base=2)})

    results = Parallel(n_jobs=6)(
        delayed(_train_ocsvm_and_score)(d, x_train_task_rep, y_test.flatten() == single_class_ind, x_test_rep)
        for d in pg)

    best_params, best_auc_score = max(zip(pg, results), key=lambda t: t[-1])
    print(best_params)
    best_ocsvm = OneClassSVM(**best_params).fit(x_train_task_rep)
    scores = best_ocsvm.decision_function(x_test_rep)
    labels = y_test.flatten() == single_class_ind

    res_file_name = '{}_cae-oc-svm_{}_{}.npz'.format(dataset_name,
                                                     get_class_name_from_index(single_class_ind, dataset_name),
                                                     datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
    save_roc_pr_curve_data(scores, labels, res_file_path)

    gpu_q.put(gpu_to_use)


def _dsebm_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q):
    gpu_to_use = gpu_q.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use

    (x_train, y_train), (x_test, y_test) = dataset_load_fn()

    n_channels = x_train.shape[get_channels_axis()]
    input_side = x_train.shape[2]  # image side will always be at shape[2]
    encoder_mdl = conv_encoder(input_side, n_channels, representation_activation='relu')
    energy_mdl = dsebm.create_energy_model(encoder_mdl)
    reconstruction_mdl = dsebm.create_reconstruction_model(energy_mdl)

    # optimization parameters
    batch_size = 128
    epochs = 200
    reconstruction_mdl.compile('adam', 'mse')
    x_train_task = x_train[y_train.flatten() == single_class_ind]
    x_test_task = x_test[y_test.flatten() == single_class_ind]  # This is just for visual monitoring
    reconstruction_mdl.fit(x=x_train_task, y=x_train_task,
                           batch_size=batch_size,
                           epochs=epochs,
                           validation_data=(x_test_task, x_test_task))

    scores = -energy_mdl.predict(x_test, batch_size)
    labels = y_test.flatten() == single_class_ind
    res_file_name = '{}_dsebm_{}_{}.npz'.format(dataset_name,
                                                get_class_name_from_index(single_class_ind, dataset_name),
                                                datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
    save_roc_pr_curve_data(scores, labels, res_file_path)

    gpu_q.put(gpu_to_use)


def _dagmm_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q):
    gpu_to_use = gpu_q.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use

    (x_train, y_train), (x_test, y_test) = dataset_load_fn()

    n_channels = x_train.shape[get_channels_axis()]
    input_side = x_train.shape[2]  # image side will always be at shape[2]
    enc = conv_encoder(input_side, n_channels, representation_dim=5,
                       representation_activation='linear')
    dec = conv_decoder(input_side, n_channels=n_channels, representation_dim=enc.output_shape[-1])
    n_components = 3
    estimation = Sequential([Dense(64, activation='tanh', input_dim=enc.output_shape[-1] + 2), Dropout(0.5),
                             Dense(10, activation='tanh'), Dropout(0.5),
                             Dense(n_components, activation='softmax')]
                            )

    batch_size = 256
    epochs = 200
    lambda_diag = 0.0005
    lambda_energy = 0.01
    dagmm_mdl = dagmm.create_dagmm_model(enc, dec, estimation, lambda_diag)
    dagmm_mdl.compile('adam', ['mse', lambda y_true, y_pred: lambda_energy*y_pred])

    x_train_task = x_train[y_train.flatten() == single_class_ind]
    x_test_task = x_test[y_test.flatten() == single_class_ind]  # This is just for visual monitoring
    dagmm_mdl.fit(x=x_train_task, y=[x_train_task, np.zeros((len(x_train_task), 1))],  # second y is dummy
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test_task, [x_test_task, np.zeros((len(x_test_task), 1))]),
                  # verbose=0
                  )

    energy_mdl = Model(dagmm_mdl.input, dagmm_mdl.output[-1])

    scores = -energy_mdl.predict(x_test, batch_size)
    scores = scores.flatten()
    if not np.all(np.isfinite(scores)):
        min_finite = np.min(scores[np.isfinite(scores)])
        scores[~np.isfinite(scores)] = min_finite - 1
    labels = y_test.flatten() == single_class_ind
    res_file_name = '{}_dagmm_{}_{}.npz'.format(dataset_name,
                                                get_class_name_from_index(single_class_ind, dataset_name),
                                                datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
    save_roc_pr_curve_data(scores, labels, res_file_path)

    gpu_q.put(gpu_to_use)


def _adgan_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q):
    gpu_to_use = gpu_q.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use

    (x_train, y_train), (x_test, y_test) = dataset_load_fn()
    if len(x_test) > 5000:
        # subsample x_test due to runtime complexity
        chosen_inds = np.random.choice(len(x_test), 5000, replace=False)
        x_test = x_test[chosen_inds]
        y_test = y_test[chosen_inds]

    n_channels = x_train.shape[get_channels_axis()]
    input_side = x_train.shape[2]  # image side will always be at shape[2]
    critic = conv_encoder(input_side, n_channels, representation_dim=1,
                          representation_activation='linear')
    noise_size = 256
    generator = conv_decoder(input_side, n_channels=n_channels, representation_dim=noise_size)

    def prior_gen(b_size):
        return np.random.normal(size=(b_size, noise_size))

    batch_size = 128
    epochs = 100

    x_train_task = x_train[y_train.flatten() == single_class_ind]

    def data_gen(b_size):
        chosen_inds = np.random.choice(len(x_train_task), b_size, replace=False)
        return x_train_task[chosen_inds]

    adgan.train_wgan_with_grad_penalty(prior_gen, generator, data_gen, critic, batch_size, epochs, grad_pen_coef=20)

    scores = adgan.scores_from_adgan_generator(x_test, prior_gen, generator)
    labels = y_test.flatten() == single_class_ind
    res_file_name = '{}_adgan_{}_{}.npz'.format(dataset_name,
                                                get_class_name_from_index(single_class_ind, dataset_name),
                                                datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
    save_roc_pr_curve_data(scores, labels, res_file_path)

    gpu_q.put(gpu_to_use)


def run_experiments(load_dataset_fn, dataset_name, q, n_classes):

    # CAE OC-SVM
    processes = [Process(target=_cae_ocsvm_experiment,
                         args=(load_dataset_fn, dataset_name, c, q)) for c in range(n_classes)]
    for p in processes:
        p.start()
        p.join()

    # Raw OC-SVM
    for c in range(n_classes):
        _raw_ocsvm_experiment(load_dataset_fn, dataset_name, c)

    n_runs = 5

    # Transformations
    for _ in range(n_runs):
        processes = [Process(target=_transformations_experiment,
                             args=(load_dataset_fn, dataset_name, c, q)) for c in range(n_classes)]
        if dataset_name in ['cats-vs-dogs']:  # Self-labeled set is memory consuming
            for p in processes:
                p.start()
                p.join()
        else:
            for p in processes:
                p.start()
            for p in processes:
                p.join()

    # DSEBM
    for _ in range(n_runs):
        processes = [Process(target=_dsebm_experiment,
                             args=(load_dataset_fn, dataset_name, c, q)) for c in range(n_classes)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    # DAGMM
    for _ in range(n_runs):
        processes = [Process(target=_dagmm_experiment,
                             args=(load_dataset_fn, dataset_name, c, q)) for c in range(n_classes)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    # ADGAN
    processes = [Process(target=_adgan_experiment,
                         args=(load_dataset_fn, dataset_name, c, q)) for c in range(n_classes)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def create_auc_table(metric='roc_auc'):
    file_path = glob(os.path.join(RESULTS_DIR, '*', '*.npz'))
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    methods = set()
    for p in file_path:
        _, f_name = os.path.split(p)
        dataset_name, method, single_class_name = f_name.split(sep='_')[:3]
        methods.add(method)
        npz = np.load(p)
        roc_auc = npz[metric]
        results[dataset_name][single_class_name][method].append(roc_auc)

    for ds_name in results:
        for sc_name in results[ds_name]:
            for method_name in results[ds_name][sc_name]:
                roc_aucs = results[ds_name][sc_name][method_name]
                results[ds_name][sc_name][method_name] = [np.mean(roc_aucs),
                                                          0 if len(roc_aucs) == 1 else scipy.stats.sem(np.array(roc_aucs))
                                                          ]

    with open('results-{}.csv'.format(metric), 'w') as csvfile:
        fieldnames = ['dataset', 'single class name'] + sorted(list(methods))
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for ds_name in sorted(results.keys()):
            for sc_name in sorted(results[ds_name].keys()):
                row_dict = {'dataset': ds_name, 'single class name': sc_name}
                row_dict.update({method_name: '{:.3f} ({:.3f})'.format(*results[ds_name][sc_name][method_name])
                                 for method_name in results[ds_name][sc_name]})
                writer.writerow(row_dict)


if __name__ == '__main__':
    freeze_support()
    N_GPUS = 2
    man = Manager()
    q = man.Queue(N_GPUS)
    for g in range(N_GPUS):
        q.put(str(g))

    experiments_list = [
        (load_cifar10, 'cifar10', 10),
        (load_cifar100, 'cifar100', 20),
        (load_fashion_mnist, 'fashion-mnist', 10),
        (load_cats_vs_dogs, 'cats-vs-dogs', 2),
    ]

    for data_load_fn, dataset_name, n_classes in experiments_list:
        run_experiments(data_load_fn, dataset_name, q, n_classes)

