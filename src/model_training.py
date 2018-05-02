import sys
import os
from time import time
from datetime import datetime
from math import sqrt

import numpy as np

import pandas as pd

from sklearn.metrics import mean_squared_error

from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.utils import multi_gpu_model

from rm_logging import time_it, logger
from rm_config import RMC
from model_config import MLC
from data_preparation import load_all_data
import model_tracking as mt


def get_tb_log_dir():
    tb_log_dir = os.path.join(RMC.TB_LOG_DIR,
                              '%s-%s-%s-%s-Y%d' % (datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H-%M-%S'),
                                                   MLC.DP, MLC.MV, MLC.OV, MLC.YEARS))

    return tb_log_dir


def apply_feature_prep_pipeline(x, fpp, fit):
    print(x.shape)
    x = x.reshape(-1, MLC.INPUT_DIM)
    print(x.shape)

    if fit:
        x = fpp.fit_transform(x)
    else:
        x = fpp.transform(x)

    print(x.shape)

    x = x.reshape(-1, MLC.INPUT_LEN, MLC.INPUT_DIM)

    return x


def execute_train(model_dir, model_file_name, fpp, model, start_epoch, end_epoch,
                  train_x, train_y, train_i, val_x, val_y, val_i):
    model_creator = MLC.get_model_creator()

    if model is None:
        fpp = model_creator.build_feature_prep_pipeline()
        model = model_creator.build_model(MLC.INPUT_LEN, MLC.INPUT_DIM, MLC.OUTPUT_DIM)
        fit = True
    else:
        fit = False

    x_t = apply_feature_prep_pipeline(x=train_x, fpp=fpp, fit=fit)
    y_t = train_y
    x_v = apply_feature_prep_pipeline(x=val_x, fpp=fpp, fit=False)
    y_v = val_y

    logger.info('Building/compiling model ...')

    if MLC.GPUS > 1:
        model = multi_gpu_model(model, gpus=MLC.GPUS)

    model = model_creator.compile_model(model)

    model_tracker = mt.Model_Tracker(model_dir, model_file_name, model, x_v, y_v, val_i)

    callbacks = [
        LearningRateScheduler(model_creator.get_learning_rate_schedule()),
        TensorBoard(log_dir=get_tb_log_dir(), histogram_freq=0, write_graph=True, write_images=False),
        model_tracker]

    if fit:
        model_tracker.save_feature_prep_pipeline(fpp)

    logger.info('Building/compiling model done.')

    logger.info('Fitting model ...')

    model.fit(
        x=[x_t], y=y_t,
        batch_size=MLC.BATCH_SIZE,
        epochs=end_epoch,
        verbose=1,
        callbacks=callbacks,
        shuffle=True,
        initial_epoch=start_epoch,
        steps_per_epoch=None,
        validation_data=[[x_v], y_v])

    logger.info('Fitting model done.')

    return fpp, model


def execute_test(fpp, model, test_x, test_y, test_i, model_dir, model_file_name):
    logger.info("Testing model ...")

    x = apply_feature_prep_pipeline(x=test_x, fpp=fpp, fit=False)
    y = test_y

    y_p = model.predict(x, verbose=1)

    y = np.reshape(a=y, newshape=(len(y),))
    y_p = np.reshape(a=y_p, newshape=(len(y),))

    test_result = pd.DataFrame(
        {RMC.SCEN_ID_COL: test_i + 1, 'y': y, 'y_pred': y_p, 'Difference': y - y_p, 'Deviation': (y - y_p) * 100 / y})
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

    for arg in sys.argv[1:]:
        if arg == 'train':
            train = True
        elif arg == 'test':
            test = True

    if not train and not test:
        train = True

    train_x, train_y, train_i, val_x, val_y, val_i, test_x, test_y, test_i = load_all_data(
        train_set=train,
        val_set=train,
        test_set=test,
        init=False)

    model_file_name = '{0}_{1}_{2}_{3}'.format(MLC.TRN, MLC.MV, MLC.OV, MLC.DP)
    model_dir = os.path.join(RMC.OUTPUT_DIR, model_file_name)

    if test or (train and not MLC.OVERWRITE):
        fpp, model = mt.load_previous_model_if_available(model_dir, model_file_name)

    if train:
        fpp, model = execute_train(model_dir, model_file_name, fpp, model,
                                   start_epoch=MLC.START_EP, end_epoch=MLC.END_EP,
                                   train_x=train_x, train_y=train_y, train_i=train_i,
                                   val_x=val_x, val_y=val_y, val_i=val_i)

    if test:
        execute_test(fpp, model, test_x, test_y, test_i, model_dir, model_file_name)

    logger.info("Main script finished in %s.", time_it(overall, time()))


if __name__ == "__main__":
    main()
