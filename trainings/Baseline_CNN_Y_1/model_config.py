from cnn_model_builders import Baseline_CNN_Model_Builder


class MLC:
    TRAIN_SIZE = 0.95
    VAL_SIZE = 0.05
    TEST_SIZE = 0.00

    YEARS = 1
    INPUT_LEN = YEARS * 12 + 1
    INPUT_DIM = 78
    OUTPUT_DIM = 1
    DP = 'DP02R00' + '_Y_' + str(YEARS)

    GPUS=1
    MV = 'MV04R01'

    BATCH_SIZE = 64
    OV = 'OV02R00'

    START_EP = 0
    END_EP = 200
    TRN = 'TR026'
    OVERWRITE = True

    MODEL_CREATOR_FILE='cnn_model_builders'

    @staticmethod
    def get_model_creator():
        return Baseline_CNN_Model_Builder()


