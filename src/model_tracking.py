import os
import pickle
from math import sqrt
from shutil import copyfile

import numpy as np
import pandas as pd

import matplotlib
from scipy import stats
from scipy.stats import binned_statistic

matplotlib.use('Agg')
from matplotlib import pyplot as plt


from keras.callbacks import Callback
from keras.models import load_model
from keras.utils import plot_model

import results_plot as rp
from riskm_config import RMC


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

        save_model_graph_and_summary(model, model_dir, model_file_name)
        save_model_source_file(model_dir, model_file_name)

        #reset history
        with open(os.path.join(self.dir, self.file_name + '_train_results_history.csv'), "w+") as file:
            file.write("loss,val_loss,mape,val_mape\n") # on change: dont forget to change content as well!


    def on_epoch_end(self, epoch, logs=None):
        loss = logs['loss']
        val_loss = logs['val_loss']
        mape = logs['mean_absolute_percentage_error']
        val_mape = logs['val_mean_absolute_percentage_error']

        #always, even when epoch is worse
        with open(os.path.join(self.dir, self.file_name + '_train_results_history.csv'), "a+") as file:
            # on change: dont forget to change headers as well!
            file.write('%.2d,%.2d, %.5f, %.5f\n' % (loss,val_loss,mape,val_mape))

        #if this epoch a better been better than the last one (or when it is the first)
        if self.best_val_loss is None or self.best_val_loss > val_loss:
            #save model
            self.best_epoch = epoch
            self.best_val_loss = val_loss

            save_keras_model(self.model, self.dir, self.file_name)

            print("New model version saved - val_rmse ({:.6f})".format(sqrt(val_loss)))

            # print model statistics
            y_p = self.model.predict(self.val_x, verbose=1)
            y_p = np.reshape(a=y_p, newshape=(len(y_p),))

            test_result = pd.DataFrame(
                {RMC.SCEN_ID_COL: self.val_i + 1, 'y': self.val_y, 'y_pred': y_p, 'Difference': self.val_y - y_p,
                 'Deviation': (self.val_y - y_p) * 100 / self.val_y})
            test_result.set_index(RMC.SCEN_ID_COL, inplace=True)
            test_result.sort_index(inplace=True)

            # skl_mse = mean_squared_error(self.val_y, y_p)
            # skl_rmse = sqrt(skl_mse)

            with open(os.path.join(self.dir, self.file_name + '_train_results_con.csv'), "w+") as file:
                # file.write("Best Epoch: {0}, Val MSE: {1}, Val RMSE: {2}\n".format(mt_callback.best_epoch, skl_mse, skl_rmse))
                # file.write("\n")
                test_result.to_csv(path_or_buf=file, columns=['y', 'y_pred', 'Difference', 'Deviation'])
                file.write(",,,, {0}\n".format(np.mean(np.absolute(self.val_y - y_p) * 100 / self.val_y)))

            # print as png
            plot(self.dir, self.file_name, self.val_y, y_p)



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


def save_model_source_file(model_dir, model_file_name):
    this_file_name = os.path.join(RMC.SRC_DIR, RMC.THIS_FILE + '.py')
    copy_file_name = os.path.join(model_dir, model_file_name + '_' + RMC.THIS_FILE + '.py')

    copyfile(this_file_name, copy_file_name)


def plot(model_dir, model_file_name, y, y_pred):

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
    fig.savefig(os.path.join(model_dir, model_file_name + '_plot.png'), dpi=100)