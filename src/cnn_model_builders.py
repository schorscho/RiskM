import math

from keras.layers import Input, Dense, Conv1D, AveragePooling1D, Flatten
from keras.models import Model

from model_builders import Model_Builder
from rm_logging import logger


class Baseline_CNN_Model_Builder(Model_Builder):
    def build_model(self, input_len, input_dim, output_dim):
        ip = Input(shape=(input_len, input_dim), name='Input_Sequence')
        print(ip.shape)
        op = Conv1D(filters=200, kernel_size=12, padding='causal', strides=1, activation='relu', name='Conv_1')(ip)
        print(op.shape)
        #op = AveragePooling1D(pool_size=2, name='Pooling_1')(op)
        print(op.shape)
        op = Conv1D(filters=100, kernel_size=6, padding='causal', strides=1, activation='relu', name='Conv_2')(op)
        print(op.shape)
        #op = AveragePooling1D(pool_size=2, name='Pooling_2')(op)
        print(op.shape)
        op = Flatten(name='Flatten')(op)
        print(op.shape)
        op = Dense(300, name='Dense_1')(op)
        op = Dense(200, name='Dense_2')(op)
        op = Dense(100, name='Dense_3')(op)
        op = Dense(output_dim, name='Prediction')(op)

        model = Model(ip, op)

        return model


 #   def get_learning_rate_schedule(self):
 #       return lr_schedule_cnn


def lr_schedule_cnn(ep):
    i_lr = 0.01
    drop = 0.5
    ep_drop = 10.0

    lr = i_lr * math.pow(drop, math.floor((1 + ep) / ep_drop))

    logger.info('New learning rate: %01.10f', lr)