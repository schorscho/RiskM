import math

from sklearn.preprocessing import StandardScaler

from rm_logging import logger


class Model_Builder():
    def build_feature_prep_pipeline(self):
        return StandardScaler()


    def build_model(self, input_len, input_dim, output_dim):
        return None


    def get_learning_rate_schedule(self):
        return lr_schedule


    def compile_model(self, model):
        # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.00, clipnorm=1.0) #epsilon=None (doesn't work)
        model.compile(optimizer='adam', loss='mse', metrics=['mape'])

        return model


def lr_schedule(ep):
    i_lr = 0.01
    drop = 0.5
    ep_drop = 10.0

    lr = i_lr * math.pow(drop, math.floor((1 + ep) / ep_drop))

    logger.info('New learning rate: %01.10f', lr)

    return lr


