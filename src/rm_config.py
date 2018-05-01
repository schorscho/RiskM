import os


class RMC:
    PROJECT_ROOT_DIR = os.environ['RM_ROOT_DIR']
    INPUT_DIR = os.path.join(PROJECT_ROOT_DIR, 'efs', 'input-data')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, 'output')
    TB_LOG_DIR= os.path.join(OUTPUT_DIR, 'tb-logs')
    SRC_DIR = os.path.join(PROJECT_ROOT_DIR, 'src')

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

    MLC_FILE='model_config'

    SCEN_ID_COL = 'SCENARIO'
    MONTH_COL = 'MONTH'



