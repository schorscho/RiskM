import sys
import os
import pickle
import logging.config
from time import time
from math import sqrt
from shutil import copyfile, rmtree

import numpy as np

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from keras.callbacks import Callback, LearningRateScheduler
from keras.layers import Input, Dense, CuDNNGRU, Bidirectional, GaussianNoise, Dropout
from keras.models import Model, load_model
from keras.utils import multi_gpu_model
from keras.utils.vis_utils import plot_model

from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as sf


DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': { 
        'standard': { 
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': { 
        'default': { 
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': { 
        'root': { 
            'handlers': ['default'],
            'level': 'INFO'
        },
        'RiskM': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        },
    } 
}


logging.config.dictConfig(DEFAULT_LOGGING)


logger = logging.getLogger('RiskM')


class RMC:
    PROJECT_ROOT_DIR = os.environ['RM_ROOT_DIR']
    INPUT_DIR = os.path.join(PROJECT_ROOT_DIR, 'input')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, 'output')
    SRC_DIR = os.path.join(PROJECT_ROOT_DIR, 'src')
    THIS_FILE = 'riskm_all'

    PROPHET_INPUT_ALL = '201709_SD_MC_EUR_Basis10k_10001_60_1'
    PROPHET_INPUT_ALL_PROPER_HEADER = '201709_SD_MC_EUR_Basis10k_10001_60_1 (proper header)'
    PROPHET_INPUT_ALL_RESHAPED = '201709_SD_MC_EUR_Basis10k_10001_60_1 (reshaped)'
    PROPHET_OUTPUT_ALL = '10k_Daten_fuer_Training_v01_fix'

    TRAIN_X_DATA_FILE = 'train_x_data'
    TRAIN_Y_DATA_FILE = 'train_y_data'
    VAL_X_DATA_FILE = 'val_x_data'
    VAL_Y_DATA_FILE = 'val_y_data'
    TEST_X_DATA_FILE = 'test_x_data'
    TEST_Y_DATA_FILE = 'test_y_data'

    SCEN_ID_COL = 'SCENARIO'
    MONTH_COL = 'MONTH'

    INPUT_LEN = 13
    INPUT_DIM = 78
    OUTPUT_DIM = 1

    TRAIN_SIZE = 0.95
    VAL_SIZE = 0.05
    TEST_SIZE = 0.00
    DP = 'DP02R00'

    GPUS=1
    MV = 'MV03R00'

    BATCH_SIZE = 256
    OV = 'OV01R00'

    START_EP = 0
    END_EP = 400
    LOAD_MODEL = 'TR010_MV03R00_OV01R00_DP02R00'
    TRN = 'TR013'
    

def build_keras_model():
    ip = Input(shape=(RMC.INPUT_LEN, RMC.INPUT_DIM), name='Input_Sequence')
    op = \
        Bidirectional(CuDNNGRU(units=300, return_sequences=True, name='RNN_1'))(ip)
    op = Dropout(0.1)(op)
    op = \
        Bidirectional(CuDNNGRU(units=300, return_sequences=True, name='RNN_2'))(op)
    op = Dropout(0.1)(op)
    op = \
        Bidirectional(CuDNNGRU(units=300, name='RNN_3'))(op)
    op = Dropout(0.1)(op)
    op = Dense(300, name='Dense_1')(op)
    op = Dropout(0.1)(op)
    op = Dense(200, name='Dense_2')(op)
    op = Dropout(0.1)(op)
    op = Dense(100, name='Dense_3')(op)
    op = Dropout(0.1)(op)
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


def time_it(start, end):
    h, r = divmod(end - start, 3600)
    m, s = divmod(r, 60)
    
    return "{:0>2}:{:0>2}:{:06.3f}".format(int(h), int(m), s)


def load_x_data(file_name, init=False):
    if init:
        data = pd.read_csv(filepath_or_buffer=os.path.join(RMC.OUTPUT_DIR, file_name + '.csv'))
    else:
        file_name += '_' + RMC.DP + '.csv'
        data = pd.read_csv(filepath_or_buffer=os.path.join(RMC.OUTPUT_DIR, file_name))
        print(data.shape)
        data.set_index([RMC.SCEN_ID_COL, RMC.MONTH_COL], inplace=True)
        print(data.shape)
        data.sort_index(inplace=True)

    return data


def load_y_data(file_name, init=False):
    if init:
        data = pd.read_csv(filepath_or_buffer=os.path.join(RMC.INPUT_DIR, file_name + '.csv'),
                           sep=';', thousands='.', decimal=',', header=0)
    else:
        file_name += '_' + RMC.DP + '.csv'
        data = pd.read_csv(filepath_or_buffer=os.path.join(RMC.OUTPUT_DIR, file_name))
        data.set_index(RMC.SCEN_ID_COL, inplace=True)
        data.sort_index(inplace=True)

    return data


def load_all_data(train_set, val_set, test_set, init):
    train_x = None
    train_y = None
    val_x = None
    val_y = None
    test_x = None
    test_y = None

    if train_set:
        logger.info("Loading training data ...")

        if init:
            train_x = load_x_data(file_name=RMC.PROPHET_INPUT_ALL, init=True)
            train_y = load_y_data(file_name=RMC.PROPHET_OUTPUT_ALL, init=True)
        else:
            train_x = load_x_data(file_name=RMC.TRAIN_X_DATA_FILE)
            train_y = load_y_data(file_name=RMC.TRAIN_Y_DATA_FILE)

        logger.info("Loading training data done.")

    if val_set:
        logger.info("Loading prepared validation data ...")

        val_x = load_x_data(file_name=RMC.VAL_X_DATA_FILE)
        val_y = load_y_data(file_name=RMC.VAL_Y_DATA_FILE)

        logger.info("Loading prepared validation data done.")

    if test_set:
        logger.info("Loading prepared test data ...")

        test_x = load_x_data(file_name=RMC.TEST_X_DATA_FILE)
        test_y = load_y_data(file_name=RMC.TEST_Y_DATA_FILE)

        logger.info("Loading prepared test data done.")

    return train_x, train_y, val_x, val_y, test_x, test_y


def save_data(data, file_name):
    file_name += '_' + RMC.DP + '.csv'

    data.to_csv(path_or_buf=os.path.join(RMC.OUTPUT_DIR, file_name))


def save_all_prepared_data(train_x, train_y, val_x, val_y, test_x, test_y):
    if train_x is not None and train_y is not None:
        logger.info("Saving prepared training data ...")

        save_data(train_x, RMC.TRAIN_X_DATA_FILE)
        save_data(train_y, RMC.TRAIN_Y_DATA_FILE)

        logger.info("Saving prepared training data done.")

    if val_x is not None and val_y is not None:
        logger.info("Saving prepared validation data ...")

        save_data(val_x, RMC.VAL_X_DATA_FILE)
        save_data(val_y, RMC.VAL_Y_DATA_FILE)

        logger.info("Saving prepared validation data done.")

    if test_x is not None and test_y is not None:
        logger.info("Saving prepared test data ...")

        save_data(test_x, RMC.TEST_X_DATA_FILE)
        save_data(test_y, RMC.TEST_Y_DATA_FILE)

        logger.info("Saving prepared test data done.")


def prepare_x_data(data):
    data.set_index(['SCENARIO', 'ECONOMY', 'CLASS', 'MEASURE', 'OS_TERM'], inplace=True)
    data = pd.DataFrame(data.stack())

    data.columns = ['VALUE']
    data.index.names = ['SCENARIO', 'ECONOMY', 'CLASS', 'MEASURE', 'OS_TERM', 'MONTH']
    data.reset_index(inplace=True)

    data['EC_CL_MS_OS'] = data['ECONOMY'] + '_' + data['CLASS'] + '_' + data['MEASURE'] + '_' + data.OS_TERM.map(str)
    data.drop(columns=['ECONOMY', 'CLASS', 'MEASURE', 'OS_TERM'], inplace=True)
    data.set_index(['SCENARIO', 'EC_CL_MS_OS', 'MONTH'], inplace=True)

    data = data.unstack(1)
    data.columns = data.columns.droplevel()

    data.sort_index(inplace=True)

    return data


def prepare_y_data(data):
    data.set_index('SCENARIO', inplace=True)
    data.sort_index(inplace=True)
    data.drop(columns='OWN_FUNDS_40', inplace=True)

    return data


def create_feature_prep_pipeline():
    return StandardScaler()


def load_feature_prep_pipeline(model_dir, model_file):
    fpp = pickle.load(open(os.path.join(model_dir, model_file + '_fpp.p'), 'rb'))

    return fpp


def save_feature_prep_pipeline(fpp, model_dir, model_file):
    pickle.dump(fpp, open(os.path.join(model_dir, model_file + '_fpp.p'), 'wb'))


def get_data_packages(x_data, y_data, fpp, fit):
    x = x_data.as_matrix()
    y = y_data.as_matrix()

    if fit:
        x = fpp.fit_transform(x)
    else:
        x = fpp.transform(x)

    x = x.reshape(-1, RMC.INPUT_LEN, RMC.INPUT_DIM)

    return x, y


def split_data(x, y, train_size, val_size, test_size):
    if val_size + test_size > 0:
        x_tr,y_tr,  x_v, y_v = split_train_test(x, y, train_size, val_size + test_size)

        if val_size == 0:
            x_te = x_v
            y_te = y_v
            x_v = None
            y_v = None
        elif test_size == 0:
            x_te = None
            y_te = None
        else:
            x_v, y_v, x_te, y_te = split_train_test(x_v, y_v,
                                                    val_size / (val_size + test_size),
                                                    test_size / (val_size + test_size))
    else:
        x_tr = x
        y_tr = y
        x_v = None
        y_v = None
        x_te = None
        y_te = None

    return x_tr, y_tr, x_v, y_v, x_te, y_te


def split_train_test(x, y, train_size, test_size):
    split = ShuffleSplit(n_splits=1, train_size=train_size, test_size=test_size, random_state=None)
    res = split.split(y)
    print(len(y))
    x_tr = None
    y_tr = None
    x_te = None
    y_te = None
    print(train_size, test_size)
    for train_i, test_i in res:
        print(np.max(train_i))
        print(np.min(train_i))
        print(len(train_i))
        x_tr = pd.DataFrame(x.loc[x.index.levels[0][train_i].values])
        y_tr = pd.DataFrame(y.iloc[train_i])
        x_te = pd.DataFrame(x.loc[x.index.levels[0][test_i].values])
        y_te = pd.DataFrame(y.iloc[test_i])

    x_tr.reset_index(inplace=True)
    x_tr.set_index(['SCENARIO', 'MONTH'], inplace=True)
    x_tr.sort_index(inplace=True)

    y_tr.sort_index(inplace=True)

    x_te.reset_index(inplace=True)
    x_te.set_index(['SCENARIO', 'MONTH'], inplace=True)
    x_te.sort_index(inplace=True)

    y_te.sort_index(inplace=True)

    return x_tr, y_tr, x_te, y_te


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


def execute_pre_init():
    logger.info("Pre-initial data preparation ...")

    then = time()

    logger.info("Creating proper header ...")

    fac = os.path.join(RMC.INPUT_DIR, RMC.PROPHET_INPUT_ALL + '.fac')
    csv = os.path.join(RMC.OUTPUT_DIR, RMC.PROPHET_INPUT_ALL_PROPER_HEADER + '.csv')

    with open(fac, 'r') as orig:
        with open(csv, 'w') as copy:
            i = 0
            for line in orig:
                if i >= 2:
                    copy.write(line)

                i += 1

                if i % 10000 == 0:
                    logger.info("Creating proper header ... %3.2f%%", (i * 100.0 / 780081))

    logger.info("Creating proper header done in %s.", time_it(then, time()))

    logger.info("Reading proper data file ...")

    then = time()
    sc = SparkContext("local[3]", "test")
    spark = SparkSession(sc)

    df = spark.read.csv(path=os.path.join(RMC.OUTPUT_DIR, RMC.PROPHET_INPUT_ALL_PROPER_HEADER + '.csv'), header=True, inferSchema=True)

    logger.info("Reading proper data file done in %s.", time_it(then, time()))

    logger.info("Collecting distinct value column names ...")

    then = time()

    df = df.withColumn('EC_CL_MS_OS', sf.concat(df.ECONOMY, sf.lit('_'), df.CLASS, sf.lit('_'),
                                                df.MEASURE, sf.lit('_'), df.OS_TERM))

    df = df.drop('!6', 'ECONOMY', 'CLASS', 'MEASURE', 'OS_TERM')

    # this is to provide faster test, needs to be uncommented later on
    #df = df.select('SCENARIO', 'EC_CL_MS_OS', '201712', '201801')

    # this is way cheaper to do now than to wait until all months have been transposed as rows
    val_col_nms = sorted(df.select('EC_CL_MS_OS').distinct().rdd.map(lambda row: row[0]).collect())

    logger.info("Collecting distinct value column names done in %s.", time_it(then, time()))

    logger.info("Transposing month columns as rows ...")

    then = time()

    keep_col_nms = ['SCENARIO', 'EC_CL_MS_OS']

    mo_col_nms = [c for c in df.columns if c not in keep_col_nms]

    mo_val_cols = sf.explode(sf.array(
        [sf.struct(sf.lit(c).alias("MONTH"), sf.col(c).alias("VALUE")) for c in mo_col_nms])).alias('MONTH_VALUE')

    df = df.select(keep_col_nms + [mo_val_cols]).select(keep_col_nms + ['MONTH_VALUE.MONTH', 'MONTH_VALUE.VALUE'])

    logger.info("Transposing month columns as rows done in %s.", time_it(then, time()))

    logger.info("Selecting with value columns ...")

    then = time()

    val_cols = [sf.when(sf.col('EC_CL_MS_OS') == c, sf.col('VALUE')).otherwise(None).alias(c)
                for c in val_col_nms]

    max_agg_cols = [sf.max(sf.col(c)).alias(c) for c in val_col_nms]

    df = df.select(sf.col('SCENARIO'), sf.col('MONTH'), *val_cols)

    logger.info("Selecting with value columns done in %s.", time_it(then, time()))

    logger.info("Aggregating value columns ...")

    then = time()

    df = df.groupBy('SCENARIO', 'MONTH').agg(*max_agg_cols)

    logger.info("Aggregating value columns done in %s.", time_it(then, time()))

    logger.info("Saving reshaped data to file ...")

    if os.path.exists(os.path.join(RMC.OUTPUT_DIR, 'tmp.csv')):
        rmtree(os.path.join(RMC.OUTPUT_DIR, 'tmp.csv'))

    df.write.csv(path=os.path.join(RMC.OUTPUT_DIR, 'tmp.csv'), mode='overwrite', header=True)

    if os.path.exists(os.path.join(RMC.OUTPUT_DIR, RMC.PROPHET_INPUT_ALL_RESHAPED + '.csv')):
        rmtree(os.path.join(RMC.OUTPUT_DIR, RMC.PROPHET_INPUT_ALL_RESHAPED + '.csv'))

    os.rename(src=os.path.join(RMC.OUTPUT_DIR, 'tmp.csv'),
                               dst=os.path.join(RMC.OUTPUT_DIR, RMC.PROPHET_INPUT_ALL_RESHAPED + '.csv'))

    logger.info("Saving reshaped data to file done in %s.", time_it(then, time()))

    sc.stop()

    logger.info("Pre-initial data preparation done in %s.", time_it(then, time()))


def execute_init(train_x, train_y):
    logger.info("Starting initial data preparation ...")

    val_x = None
    val_y = None
    test_x = None
    test_y = None

    if train_x is not None and train_y is not None:
        logger.info("Preparing training data ...")

        train_x = prepare_x_data(train_x)
        train_y = prepare_y_data(train_y)

        logger.info("Preparing training data done.")

        logger.info("Splitting prepared training data ...")

        train_x_as_test = None
        train_y_as_test = None

        if RMC.TEST_SIZE == 0:
            train_x_as_test = train_x
            train_y_as_test = train_y

        train_x, train_y, val_x, val_y, test_x, test_y = split_data(x=train_x, y=train_y,
                                                                    train_size=RMC.TRAIN_SIZE,
                                                                    val_size=RMC.VAL_SIZE,
                                                                    test_size=RMC.TEST_SIZE)

        if RMC.TEST_SIZE == 0:
            test_x = train_x_as_test
            test_y = train_y_as_test

        logger.info("Splitting prepared training data done.")

    return train_x, train_y, val_x, val_y, test_x, test_y


def execute_train(model_dir, model_file_name, start_epoch, end_epoch, fpp, build_on_model, train_x, train_y, val_x, val_y):
    if fpp is None:
        fpp = create_feature_prep_pipeline()
        fit = True
    else:
        fit = False

    x_t, y_t = get_data_packages(x_data=train_x, y_data=train_y, fpp=fpp, fit=fit)
    x_v, y_v = get_data_packages(x_data=val_x, y_data=val_y, fpp=fpp, fit=False)

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

    x, y = get_data_packages(x_data=test_x, y_data=test_y, fpp=fpp, fit=False)

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

    pre_init = False
    init = False
    train = False
    test = False

    fpp = None
    model = None
    model_file_name = None
    model_dir = None

    for arg in sys.argv[1:]:
        if arg == 'pre_init':
            pre_init = True
        if arg == 'init':
            initialization = True
        elif arg == 'train':
            train = True
        elif arg == 'test':
            test = True

    if not pre_init and not init and not train and not test:
        init = True
        train = True

    if pre_init:
        execute_pre_init()

    train_x, train_y, val_x, val_y, test_x, test_y = load_all_data(
        train_set=(train or init),
        val_set=(train and not init),
        test_set=(test and not init),
        init=init)

    if init:
        train_x, train_y, val_x, val_y, test_x, test_y = execute_init(
            train_x=train_x, train_y=train_y)

        save_all_prepared_data(
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            test_x=test_x,
            test_y=test_y)

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
