import os
import pickle
from math import sqrt
from shutil import copyfile

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import binned_statistic

from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from keras.callbacks import Callback
from keras.models import load_model
from keras.utils import plot_model

from rm_logging import logger
from rm_config import RMC
from model_config import MLC


class Model_Tracker(Callback):
    def __init__(self, model_dir, model_file_name, model, val_x, val_y, val_i):
        super(Callback, self).__init__()

        self.model = model
        self.file_name = model_file_name
        self.dir = model_dir
        self.best_epoch = None
        self.best_val_loss = None

        self.val_x = val_x
        self.val_y = np.reshape(a=val_y, newshape=(len(val_y),))
        self.val_i = val_i

        # This is because we deleted scenario 2053 (index 2052 in numpy array) from data set
        self.val_i[val_i > 2051] += 1

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.save_model_graph_and_summary()
        self.save_model_source_files()

        #reset history
        self.track_training_history(init=True)


    def on_epoch_end(self, epoch, logs=None):
        loss = logs['loss']
        val_loss = logs['val_loss']
        mape = logs['mean_absolute_percentage_error']
        val_mape = logs['val_mean_absolute_percentage_error']

        #always, even when epoch is worse
        self.track_training_history(epoch + 1, loss, val_loss, mape, val_mape)
        self.plot_training_history()

        #if this epoch a better been better than the last one (or when it is the first)
        if self.best_val_loss is None or self.best_val_loss > val_loss:
            #save model
            self.best_epoch = epoch + 1
            self.best_val_loss = val_loss

            self.save_keras_model()

            print("New model version saved - val_rmse ({:.6f})".format(sqrt(val_loss)))

            y_p = self.model.predict(self.val_x, verbose=1)
            y_p = np.reshape(a=y_p, newshape=(len(y_p),))

            self.save_validation_results(self.val_i, self.val_y, y_p)
            self.plot_validation_results(self.val_y, y_p)


    def track_training_history(self, epoch=None, loss=None, val_loss=None, mape=None, val_mape=None, init=False):
        mode = 'w+' if init else 'a+'

        with open(os.path.join(self.dir, self.file_name + '_training_history.csv'), mode) as file:
            if init:
                file.write("epoch,loss,val_loss,mape,val_mape\n") # on change: dont forget to change content as well!
            else:
                # on change: dont forget to change headers as well!
                file.write('%s,%.2d,%.2d,%.5f,%.5f\n' % (str(epoch), loss, val_loss, mape, val_mape))


    def plot_training_history(self):
        hist = pd.read_csv(filepath_or_buffer=os.path.join(self.dir, self.file_name + '_training_history.csv'),
                         header=0)

        plt.plot(hist['loss'])
        plt.plot(hist['val_loss'])
        plt.yscale('log')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')

        fig = plt.gcf()
        fig.set_size_inches(12, 8)
        fig.savefig(os.path.join(self.dir, self.file_name + '_training_history.png'), dpi=100)

        plt.close()


    def save_validation_results(self, i, y, y_pred):
        val_result = pd.DataFrame(
            {RMC.SCEN_ID_COL: i + 1, 'Y': y, 'Y_PRED': y_pred, 'ERROR': y - y_pred,
             'PERCENTAGE_ERROR': (y - y_pred) * 100 / y})
        val_result.set_index(RMC.SCEN_ID_COL, inplace=True)
        val_result.sort_index(inplace=True)
        val_result.to_csv(path_or_buf=os.path.join(self.dir, self.file_name + '_validation_results.csv'))

        skl_mse = mean_squared_error(y, y_pred)
        skl_rmse = sqrt(skl_mse)

        with open(os.path.join(self.dir, self.file_name + '_validation_results_summary.csv'), "w+") as file:
            file.write("Best Epoch: {0}\n".format(self.best_epoch))
            file.write("Validation MSE: {0}\n".format(skl_mse))
            file.write("Validation RMSE: {0}\n".format(skl_rmse))
            file.write("Validation MAPE: {0}\n".format(np.mean(np.absolute(y - y_pred) * 100 / y)))


    def plot_validation_results(self, y, y_pred):
        plt.figure(figsize=(16,20))
        plt.subplots_adjust(hspace=0.4)
        ax1 = plt.subplot(411)
        plt.hist(y, density=True)
        plt.title('Distribution of y (Prophet)')
        plt.xlabel('y (Prophet)')
        plt.ylabel('Number of Samples')
        xt = plt.xticks()[0]
        xmin, xmax = min(xt), max(xt)
        lnspc = np.linspace(xmin, xmax, len(y))
        m, s = stats.norm.fit(y)
        pdf_g = stats.norm.pdf(lnspc, m, s)
        plt.plot(lnspc, pdf_g, label='Norm')

        plt.subplot(412, sharex=ax1)
        plt.title('Distribution of y_pred (ANN)')
        plt.xlabel('y_pred (ANN)')
        plt.ylabel('Number of Samples')
        plt.hist(y_pred, density=True)
        xt = plt.xticks()[0]
        xmin, xmax = min(xt), max(xt)
        lnspc = np.linspace(xmin, xmax, len(y))
        m, s = stats.norm.fit(y_pred)
        pdf_g = stats.norm.pdf(lnspc, m, s)
        plt.plot(lnspc, pdf_g, label='Norm')

        plt.subplot(413, sharex=ax1)
        plt.title('Absolute Errors of ANN over y (Prophet)')
        plt.xlabel('y (Prophet)')
        plt.ylabel('Error')
        plt.scatter(x=y, y=np.abs(y - y_pred), s=3)

        plt.subplot(414, sharex=ax1)
        deviation = np.abs(y - y_pred)
        bin_means, bin_edges, binnumber = binned_statistic(x=y, values=deviation, statistic='mean', bins=10)

        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width/2

        occ = np.bincount(binnumber)
        occ = np.delete(occ, 0)

        plt.title('Mean Absolute Errors of ANN in Y-Bins (size indicates number of samples)')
        plt.xlabel('y (Prophet)')
        plt.ylabel('Error')
        plt.scatter(x=bin_centers, y=bin_means, s=occ * 3)

        plt.subplots_adjust(top=0.85)

        fig = plt.gcf()
        # fig.tight_layout(rect=[0.01, 0.2, 1, 0.9])
        fig.set_size_inches(12, 10)
        fig.savefig(os.path.join(self.dir, self.file_name + '_validation_results.png'), dpi=100)

        plt.close()


    def save_feature_prep_pipeline(self, fpp):
        pickle.dump(fpp, open(os.path.join(self.dir, self.file_name + '_fpp.p'), 'wb'))


    def save_keras_model(self):
        self.model.save(os.path.join(self.dir, self.file_name + '_model.h5'))


    def save_model_graph_and_summary(self):
        plot_model(self.model, to_file=os.path.join(self.dir, self.file_name + '_model.png'), show_shapes=True)

        with open(os.path.join(self.dir, self.file_name + '_model.txt'), 'w') as fh:
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))


    def save_model_source_files(self):
        orig_file_name = os.path.join(RMC.SRC_DIR, RMC.MLC_FILE + '.py')
        copy_file_name = os.path.join(self.dir, RMC.MLC_FILE + '.py')

        copyfile(orig_file_name, copy_file_name)

        orig_file_name = os.path.join(RMC.SRC_DIR, MLC.MODEL_CREATOR_FILE + '.py')
        copy_file_name = os.path.join(self.dir, MLC.MODEL_CREATOR_FILE + '.py')

        copyfile(orig_file_name, copy_file_name)


def previous_keras_model_file_exists(model_dir, model_file_name):
    return os.path.exists(os.path.join(model_dir, model_file_name + '_model.h5'))


def load_feature_prep_pipeline(model_dir, model_file_name):
    fpp = pickle.load(open(os.path.join(model_dir, model_file_name + '_fpp.p'), 'rb'))

    return fpp


def load_keras_model(model_dir, model_file_name):
    model = load_model(os.path.join(model_dir, model_file_name + '_model.h5'))

    return model


def load_previous_model_if_available(model_dir, model_file_name):
    fpp = None
    model = None

    if previous_keras_model_file_exists(model_dir, model_file_name):
        logger.info("Loading model ...")

        fpp = load_feature_prep_pipeline(model_dir, model_file_name)
        model = load_keras_model(model_dir, model_file_name)

        logger.info("Loading model done.")

    return fpp, model
