import sys
import os
import pickle
import logging.config
from time import time
from math import sqrt
from shutil import copyfile

import numpy as np

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from keras.callbacks import Callback, LearningRateScheduler
from keras.layers import Input, Dense, CuDNNGRU, Bidirectional, GaussianNoise, Dropout
from keras.models import Model, load_model
from keras.utils import multi_gpu_model
from keras.utils.vis_utils import plot_model

from riskm_config import RMC, time_it, logger
from data_preparation import load_all_data


def build_keras_model():
    ip = Input(shape=(RMC.INPUT_LEN, RMC.INPUT_DIM), name='Input_Sequence')
    op = Bidirectional(CuDNNGRU(units=600, return_sequences=True, name='RNN_1'))(ip)
    op = Bidirectional(CuDNNGRU(units=600, return_sequences=True, name='RNN_2'))(op)
    op = Bidirectional(CuDNNGRU(units=600, name='RNN_3'))(op)
    op = Dense(600, name='Dense_1')(op)
    op = Dense(300, name='Dense_2')(op)
    op = Dense(200, name='Dense_3')(op)
    op = Dense(1, name='Prediction')(op)

    model = Model(ip, op)

    return model


def lr_schedule(ep):
    lr = 0.001

    lr = lr / (ep // 10 + 1)

    logger.info('New learning rate: %01.10f', lr)

    return lr


def compile_keras_model(model):
    #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.00, clipnorm=1.0) #epsilon=None (doesn't work)
    if RMC.GPUS > 1:
        model = multi_gpu_model(model, gpus=RMC.GPUS)

    model.compile(optimizer='adam', loss='mse', metrics=['mape'])

    return model


def create_feature_prep_pipeline():
    return StandardScaler()


def apply_feature_prep_pipeline(x, fpp, fit):
    print(x.shape)
    x = x.reshape(-1, RMC.INPUT_DIM)
    print(x.shape)
    if fit:
        x = fpp.fit_transform(x)
    else:
        x = fpp.transform(x)

    print(x.shape)
    x = x.reshape(-1, RMC.INPUT_LEN, RMC.INPUT_DIM)
    print(x.shape)

    return x


def load_feature_prep_pipeline(model_dir, model_file):
    fpp = pickle.load(open(os.path.join(model_dir, model_file + '_fpp.p'), 'rb'))

    return fpp


def save_feature_prep_pipeline(fpp, model_dir, model_file):
    pickle.dump(fpp, open(os.path.join(model_dir, model_file + '_fpp.p'), 'wb'))


def previous_keras_model_file_exists(model_dir, model_file_name):
    return os.path.exists(os.path.join(model_dir, model_file_name + '_model.h5'))


def load_keras_model(model_dir, model_file_name):
    model = load_model(os.path.join(model_dir, model_file_name + '_model.h5'))

    return model


def save_keras_model(model, model_dir, model_file_name):
    model.save(os.path.join(model_dir, model_file_name + '_model.h5'))
    

def save_training_history(history, model_dir, model_file_name):
    hist = pd.DataFrame.from_dict(history.history)
    hist['epoch'] = [i + 1 for i in range(len(hist))]
    hist.set_index('epoch', inplace=True)
    hist.to_csv(path_or_buf=os.path.join(model_dir, model_file_name + '_history.csv'))

    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.yscale('log')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')

    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    fig.savefig(os.path.join(model_dir, model_file_name + '_history.png'), dpi=100)


def save_model_graph_and_summary(model, model_dir, model_file_name):
    plot_model(model, to_file=os.path.join(model_dir, model_file_name + '_model.png'), show_shapes=True)

    with open(os.path.join(model_dir, model_file_name + '_model.txt'), 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


def copy_this_file(model_dir, model_file_name):
    this_file_name = os.path.join(RMC.SRC_DIR, RMC.THIS_FILE + '.py')
    copy_file_name = os.path.join(model_dir, model_file_name + '_' + RMC.THIS_FILE + '.py')

    copyfile(this_file_name, copy_file_name)


class Model_Tracker(Callback):
    def __init__(self, model_dir, model_file_name, model):
        super(Callback, self).__init__()

        self.model = model
        self.file_name = model_file_name
        self.dir = model_dir
        self.best_epoch = None
        self.best_val_loss = None


    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']

        if self.best_val_loss is None or self.best_val_loss > val_loss:
            self.best_epoch = epoch
            self.best_val_loss = val_loss

            save_keras_model(self.model, self.dir, self.file_name)

            print("New model version saved - val_rmse ({:.6f})".format(sqrt(val_loss)))


def execute_train(model_dir, model_file_name, start_epoch, end_epoch, fpp, build_on_model, train_x, train_y, val_x, val_y):
    if fpp is None:
        fpp = create_feature_prep_pipeline()
        fit = True
    else:
        fit = False

    x_t = apply_feature_prep_pipeline(x=train_x, fpp=fpp, fit=fit)
    y_t = train_y
    x_v = apply_feature_prep_pipeline(x=val_x, fpp=fpp, fit=False)
    y_v = val_y

    logger.info('Building/compiling model ...')

    if build_on_model is None:
        model = build_keras_model()
        model = compile_keras_model(model)
    else:
        model = build_on_model

    callbacks = [LearningRateScheduler(lr_schedule)]

    mt_callback = None

    if model_file_name is not None:
        mt_callback = Model_Tracker(model_dir, model_file_name, model=model)

        callbacks.append(mt_callback)

        save_model_graph_and_summary(model, model_dir, model_file_name)
        save_feature_prep_pipeline(fpp, model_dir, model_file_name)
        copy_this_file(model_dir, model_file_name)

    logger.info('Building/compiling model done.')

    logger.info('Fitting model ...')

    history = model.fit(
        x=[x_t], y=y_t,
        batch_size=RMC.BATCH_SIZE,
        epochs=end_epoch,
        verbose=1,
        callbacks=callbacks,
        shuffle=True,
        initial_epoch=start_epoch,
        steps_per_epoch=None,
        validation_data=[[x_v], y_v])

    if model_file_name is not None:
        save_training_history(history, model_dir, model_file_name)

    y_p = model.predict(x_v, verbose=1)

    y = np.reshape(a=y_v, newshape=(len(y_v),))
    y_p = np.reshape(a=y_p, newshape=(len(y_v),))

    test_result = pd.DataFrame(
        {RMC.SCEN_ID_COL: val_x.index.levels[0], 'y': y, 'y_pred': y_p, 'Difference': y - y_p, 'Deviation': (y - y_p) * 100 / y})
    test_result.set_index(RMC.SCEN_ID_COL, inplace=True)
    test_result.sort_index(inplace=True)

    skl_mse = mean_squared_error(y, y_p)
    skl_rmse = sqrt(skl_mse)

    if model_file_name is not None:
        with open(os.path.join(model_dir, model_file_name + '_train_results.csv'), "w") as file:
            file.write("Best Epoch: {0}, Val MSE: {1}, Val RMSE: {2}\n".format(mt_callback.best_epoch, skl_mse, skl_rmse))
            file.write("\n")
            test_result.to_csv(path_or_buf=file, columns=['y', 'y_pred', 'Difference', 'Deviation'])
            file.write(",,,, {0}\n".format(np.mean(np.absolute(y - y_p) * 100 / y)))

    logger.info('Fitting model done.')

    return fpp, model


def execute_test(fpp, model, test_x, test_y, model_dir, model_file_name):
    logger.info("Testing model ...")

    x = apply_feature_prep_pipeline(x=test_x, fpp=fpp, fit=False)
    y = test_y

    y_p = model.predict(x, verbose=1)

    y = np.reshape(a=y, newshape=(len(y),))
    y_p = np.reshape(a=y_p, newshape=(len(y),))

    test_result = pd.DataFrame(
        {RMC.SCEN_ID_COL: test_x.index.levels[0], 'y': y, 'y_pred': y_p, 'Difference': y - y_p, 'Deviation': (y - y_p) * 100 / y})
    test_result.set_index(RMC.SCEN_ID_COL, inplace=True)
    test_result.sort_index(inplace=True)

    skl_mse = mean_squared_error(y, y_p)
    skl_rmse = sqrt(skl_mse)

    print(" - test_skl_mse ({:.6f}), test_skl_rmse ({:.6f})".format(skl_mse, skl_rmse))
    print('\n')

    if model_dir is not None:
        with open(os.path.join(model_dir, model_file_name + '_test_results.csv'), "w") as file:
            file.write("Test MSE: {0}, Test RMSE: {1}\n".format(skl_mse, skl_rmse))
            file.write("\n")
            test_result.to_csv(path_or_buf=file, columns=['y', 'y_pred', 'Difference', 'Deviation'])
            file.write(",,,, {0}\n".format(np.mean(np.absolute(y - y_p) * 100 / y)))


def main():
    overall = time()

    logger.info("Main script started ...")

    train = False
    test = False

    fpp = None
    model = None
    model_file_name = None
    model_dir = None

    for arg in sys.argv[1:]:
        if arg == 'train':
            train = True
        elif arg == 'test':
            test = True

    if not train and not test:
        train = True

    train_x, train_y, val_x, val_y, test_x, test_y = load_all_data(
        train_set=train,
        val_set=train,
        test_set=test,
        init=False)

    if train or test:
        if RMC.TRN is not None:
            model_file_name = '{0}_{1}_{2}_{3}'.format(RMC.TRN, RMC.MV, RMC.OV, RMC.DP)
            model_dir = os.path.join(RMC.OUTPUT_DIR, model_file_name)

            if not os.path.exists(model_dir) and train:
                os.makedirs(model_dir)

            if previous_keras_model_file_exists(model_dir, model_file_name):
                logger.info("Loading model ...")

                fpp = load_feature_prep_pipeline(model_dir, model_file_name)
                model = load_keras_model(model_dir, model_file_name)

                logger.info("Loading model done.")

    if train:
        fpp, model = execute_train(model_dir, model_file_name,
                                   start_epoch=RMC.START_EP, end_epoch=RMC.END_EP,
                                   fpp=fpp, build_on_model=model,
                                   train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)

    if test:
        execute_test(fpp, model, test_x, test_y, model_dir, model_file_name)

    logger.info("Main script finished in %s.", time_it(overall, time()))


if __name__ == "__main__":
    main()
