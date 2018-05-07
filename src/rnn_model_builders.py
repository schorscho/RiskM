from keras.layers import Input, Dense, CuDNNGRU, Bidirectional, Reshape
from keras.models import Model

from model_builders import Model_Builder


class Baseline_RNN_Model_Builder(Model_Builder):
    def build_model(self, input_len, input_dim, output_dim):
        ip = Input(shape=(input_len, input_dim), name='Input_Sequence')
        op = Bidirectional(CuDNNGRU(units=300, return_sequences=True, name='RNN_1'))(ip)
        op = Bidirectional(CuDNNGRU(units=300, return_sequences=True, name='RNN_2'))(op)
        op = Bidirectional(CuDNNGRU(units=300, name='RNN_3'))(op)
        op = Dense(300, name='Dense_1')(op)
        op = Dense(200, name='Dense_2')(op)
        op = Dense(100, name='Dense_3')(op)
        op = Dense(output_dim, name='Prediction')(op)

        model = Model(ip, op)

        return model


class AdaBoost_RNN_Model_Builder(Model_Builder):
    def build_model(self, input_len, input_dim, output_dim):
        ip = Input(shape=(input_len * input_dim,), name='Input_Sequence')
        print(ip.shape)
        op = Reshape(target_shape=(input_len, input_dim), name='Reshape')(ip)
        print(op.shape)
        op = Bidirectional(CuDNNGRU(units=300, return_sequences=True, name='RNN_1'))(op)
        op = Bidirectional(CuDNNGRU(units=300, return_sequences=True, name='RNN_2'))(op)
        op = Bidirectional(CuDNNGRU(units=300, name='RNN_3'))(op)
        op = Dense(300, name='Dense_1')(op)
        op = Dense(200, name='Dense_2')(op)
        op = Dense(100, name='Dense_3')(op)
        op = Dense(output_dim, name='Prediction')(op)

        model = Model(ip, op)

        model.compile(optimizer='adam', loss='mse', metrics=['mape'])

        return model
