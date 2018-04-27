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
from keras.layers import Input, Dense, CuDNNGRU, Bidirectional, Reshape
from keras.models import Model
from keras.utils import multi_gpu_model

from riskm_config import RMC, time_it, logger
from data_preparation import load_all_data
import model_tracking as mt


def build_keras_model():
    ip = Input(shape=(RMC.INPUT_LEN, RMC.INPUT_DIM), name='Input_Sequence')
    op = Bidirectional(CuDNNGRU(units=300, return_sequences=True, name='RNN_1'))(ip)
    op = Bidirectional(CuDNNGRU(units=300, return_sequences=True, name='RNN_2'))(op)
    op = Bidirectional(CuDNNGRU(units=300, name='RNN_3'))(op)
    op = Dense(300, name='Dense_1')(op)
    op = Dense(200, name='Dense_2')(op)
    op = Dense(100, name='Dense_3')(op)
    op = Dense(1, name='Prediction')(op)

    model = Model(ip, op)

    return model

def lr_schedule(ep):
    i_lr = 0.01
    drop = 0.5
    ep_drop = 10.0

    lr = i_lr * math.pow(drop, math.floor((1 + ep) / ep_drop))

    logger.info('New learning rate: %01.10f', lr)

    return lr


def compile_keras_model(model):
    #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.00, clipnorm=1.0) #epsilon=None (doesn't work)
    if RMC.GPUS > 1:
        model = multi_gpu_model(model, gpus=RMC.GPUS)

    model.compile(optimizer='adam', loss='mse', metrics=['mape'])

    return model


