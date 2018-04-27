import sys
import os
import math
from time import time
from math import sqrt

import numpy as np

import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from keras.callbacks import LearningRateScheduler, TensorBoard

from riskm_config import RMC, time_it, logger
from data_preparation import load_all_data
import model_tracking as mt
import model_creation as mc


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

    return x


def execute_train(model_dir, model_file_name, start_epoch, end_epoch, fpp, build_on_model,
                  train_x, train_y, train_i, val_x, val_y, val_i):
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
        model = mc.build_keras_model()
        model = mc.compile_keras_model(model)
    else:
        model = build_on_model

    tbCallback = TensorBoard(log_dir=RMC.tb_log_dir, histogram_freq=0, write_graph=True, write_images=False)
    callbacks = [
        LearningRateScheduler(mc.lr_schedule),
        tbCallback]


    if model_file_name is not None:
        mt_callback = mt.Model_Tracker(model_dir, model_file_name, model, val_x, val_y, val_i)

        callbacks.append(mt_callback)

        mt.save_feature_prep_pipeline(fpp, model_dir, model_file_name)

    logger.info('Building/compiling model done.')

    logger.info('Fitting model ...')

    model.fit(
        x=[x_t], y=y_t,
        batch_size=RMC.BATCH_SIZE,
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
    model_file_name = None
    model_dir = None

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

    if train or test:
        if RMC.TRN is not None:
            model_file_name = '{0}_{1}_{2}_{3}'.format(RMC.TRN, RMC.MV, RMC.OV, RMC.DP)
            model_dir = os.path.join(RMC.OUTPUT_DIR, model_file_name)

            if not os.path.exists(model_dir) and train:
                os.makedirs(model_dir)

            if mt.previous_keras_model_file_exists(model_dir, model_file_name):
               logger.info("Loading model ...")

               fpp = mt.load_feature_prep_pipeline(model_dir, model_file_name)
               model = mt.load_keras_model(model_dir, model_file_name)

               logger.info("Loading model done.")

    if train:
        fpp, model = execute_train(model_dir, model_file_name,
                                   start_epoch=RMC.START_EP, end_epoch=RMC.END_EP,
                                   fpp=fpp, build_on_model=model,
                                   train_x=train_x, train_y=train_y, train_i=train_i,
                                   val_x=val_x, val_y=val_y, val_i=val_i)

    if test:
        execute_test(fpp, model, test_x, test_y, test_i, model_dir, model_file_name)

    logger.info("Main script finished in %s.", time_it(overall, time()))


if __name__ == "__main__":
    main()
