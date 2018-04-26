import os
import logging.config


DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': { 
        'standard': { 
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': { 
        'default': { 
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': { 
        'root': { 
            'handlers': ['default'],
            'level': 'INFO'
        },
        'RiskM': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        },
    } 
}


logging.config.dictConfig(DEFAULT_LOGGING)


logger = logging.getLogger('RiskM')


class RMC:
    PROJECT_ROOT_DIR = os.environ['RM_ROOT_DIR']
    INPUT_DIR = os.path.join(PROJECT_ROOT_DIR, 'input')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, 'output')
    SRC_DIR = os.path.join(PROJECT_ROOT_DIR, 'src')
    THIS_FILE = 'riskm_full'

    PROPHET_INPUT_ALL = '201709_SD_MC_EUR_Basis10k_10001_60_1'
    PROPHET_INPUT_ALL_PROPER = '201709_SD_MC_EUR_Basis10k_10001_60_1_(proper)'
    PROPHET_INPUT_ALL_RESHAPED = '201709_SD_MC_EUR_Basis10k_10001_60_1_(reshaped)'
    PROPHET_INPUT_ALL_NUMPY = '201709_SD_MC_EUR_Basis10k_10001_60_1_(numpy)'
    PROPHET_OUTPUT_ALL = '10k_Daten_fuer_Training_v01_fix_(all_40_years)'
    PROPHET_OUTPUT_ALL_PROPER = '10k_Daten_fuer_Training_v01_fix_(proper)'
    PROPHET_OUTPUT_ALL_NUMPY = '10k_Daten_fuer_Training_v01_fix_(numpy)'

    TRAIN_X_DATA_FILE = 'train_x_data'
    TRAIN_Y_DATA_FILE = 'train_y_data'
    TRAIN_I_DATA_FILE = 'train_i_data'
    VAL_X_DATA_FILE = 'val_x_data'
    VAL_Y_DATA_FILE = 'val_y_data'
    VAL_I_DATA_FILE = 'val_i_data'
    TEST_X_DATA_FILE = 'test_x_data'
    TEST_Y_DATA_FILE = 'test_y_data'
    TEST_I_DATA_FILE = 'test_i_data'

    SCEN_ID_COL = 'SCENARIO'
    MONTH_COL = 'MONTH'

    TRAIN_SIZE = 0.95
    VAL_SIZE = 0.05
    TEST_SIZE = 0.00
    YEARS = 40
    INPUT_LEN = YEARS * 12 + 1
    INPUT_DIM = 78
    OUTPUT_DIM = 1
    DP = 'DP02R00'

    GPUS=1
    MV = 'MV04R00'

    BATCH_SIZE = 64
    OV = 'OV01R02'

    START_EP = 0
    END_EP = 60
    LOAD_MODEL = None
    TRN = 'TR018'


def time_it(start, end):
    h, r = divmod(end - start, 3600)
    m, s = divmod(r, 60)
    
    return "{:0>2}:{:0>2}:{:06.3f}".format(int(h), int(m), s)


