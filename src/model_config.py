from rnn_model_builders import Baseline_RNN_Model_Builder, AdaBoost_RNN_Model_Builder


class MLC:
    TRAIN_SIZE = 0.95
    VAL_SIZE = 0.05
    TEST_SIZE = 0.00

    YEARS = 40
    INPUT_LEN = YEARS * 12 + 1
    INPUT_DIM = 78
    OUTPUT_DIM = 1
    DP = 'DP02R00' + '_Y_' + str(YEARS)

    GPUS=1
    MV = 'MV04R01'

    BATCH_SIZE = 64
    OV = 'OV02R00'

    START_EP = 0
    END_EP = 10
    TRN = 'TR030'
    OVERWRITE = False

    MODEL_CREATOR_FILE='rnn_model_builders'

    @staticmethod
    def get_model_creator():
        return Baseline_RNN_Model_Builder()
        #return AdaBoost_RNN_Model_Builder()


