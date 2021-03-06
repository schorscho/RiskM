import sys
import os
from time import time
from shutil import rmtree

import numpy as np

from sklearn.model_selection import ShuffleSplit

from rm_logging import logger, time_it
from rm_config import RMC
from model_config import MLC


def load_data(file_name, init=False):
    if init:
        file_name = file_name + '.npy'
    else:
        file_name += '_' + MLC.DP + '.npy'

    data = np.load(os.path.join(RMC.OUTPUT_DIR, file_name))

    return data


def load_all_data(train_set, val_set, test_set, init):
    train_x = None
    train_y = None
    train_i = None
    val_x = None
    val_y = None
    val_i = None
    test_x = None
    test_y = None
    test_i = None

    if train_set:
        logger.info("Loading training data ...")

        if init:
            train_x = load_data(file_name=RMC.PROPHET_INPUT_ALL_NUMPY, init=True)
            train_y = load_data(file_name=RMC.PROPHET_OUTPUT_ALL_NUMPY, init=True)
        else:
            train_x = load_data(file_name=RMC.TRAIN_X_DATA_FILE)
            train_y = load_data(file_name=RMC.TRAIN_Y_DATA_FILE)
            train_i = load_data(file_name=RMC.TRAIN_I_DATA_FILE)

        logger.info("Loading training data done.")

    if val_set:
        logger.info("Loading prepared validation data ...")

        val_x = load_data(file_name=RMC.VAL_X_DATA_FILE)
        val_y = load_data(file_name=RMC.VAL_Y_DATA_FILE)
        val_i = load_data(file_name=RMC.VAL_I_DATA_FILE)

        logger.info("Loading prepared validation data done.")

    if test_set:
        logger.info("Loading prepared test data ...")

        test_x = load_data(file_name=RMC.TEST_X_DATA_FILE)
        test_y = load_data(file_name=RMC.TEST_Y_DATA_FILE)
        test_i = load_data(file_name=RMC.TEST_I_DATA_FILE)

        logger.info("Loading prepared test data done.")

    return train_x, train_y, train_i, val_x, val_y, val_i, test_x, test_y, test_i


def save_data(data, file_name):
    file_name += '_' + MLC.DP + '.npy'

    np.save(file=os.path.join(RMC.OUTPUT_DIR, file_name), arr=data)


def save_all_data(train_x, train_y, train_i, val_x, val_y, val_i, test_x, test_y, test_i):
    if train_x is not None and train_y is not None:
        logger.info("Saving prepared training data ...")

        save_data(train_x, RMC.TRAIN_X_DATA_FILE)
        save_data(train_y, RMC.TRAIN_Y_DATA_FILE)
        save_data(train_i, RMC.TRAIN_I_DATA_FILE)

        logger.info("Saving prepared training data done.")

    if val_x is not None and val_y is not None:
        logger.info("Saving prepared validation data ...")

        save_data(val_x, RMC.VAL_X_DATA_FILE)
        save_data(val_y, RMC.VAL_Y_DATA_FILE)
        save_data(val_i, RMC.VAL_I_DATA_FILE)

        logger.info("Saving prepared validation data done.")

    if test_x is not None and test_y is not None:
        logger.info("Saving prepared test data ...")

        save_data(test_x, RMC.TEST_X_DATA_FILE)
        save_data(test_y, RMC.TEST_Y_DATA_FILE)
        save_data(test_i, RMC.TEST_I_DATA_FILE)

        logger.info("Saving prepared test data done.")


def trim_data(x_data, y_data, years):
    x_data = x_data[:,0:12 * years + 1,:]
    y_data = y_data[:,years - 1]
    return x_data, y_data


def split_data(x, y, train_size, val_size, test_size):
    if val_size + test_size > 0:
        x_tr,y_tr, x_v, y_v, i_tr, i_v = split_train_test(x, y, train_size, val_size + test_size)

        if val_size == 0:
            x_te = x_v
            y_te = y_v
            i_te = i_v
            x_v = None
            y_v = None
            i_v = None
        elif test_size == 0:
            x_te = None
            y_te = None
            i_te = None
        else:
            x_v, y_v, x_te, y_te, i_v, i_te = split_train_test(x_v, y_v,
                                                    val_size / (val_size + test_size),
                                                    test_size / (val_size + test_size))
    else:
        x_tr = x
        y_tr = y
        i_tr = range(len(y_tr))
        x_v = None
        y_v = None
        i_v = None
        x_te = None
        y_te = None
        i_te = None

    return x_tr, y_tr, x_v, y_v, x_te, y_te, i_tr, i_v, i_te


def split_train_test(x, y, train_size, test_size):
    split = ShuffleSplit(n_splits=1, train_size=train_size, test_size=test_size, random_state=None)
    res = split.split(y)
    x_tr = None
    y_tr = None
    x_te = None
    y_te = None
    train_i = None
    test_i = None

    for train_i, test_i in res:
        x_tr = x[train_i]
        y_tr = y[train_i]
        x_te = x[test_i]
        y_te = y[test_i]

    return x_tr, y_tr, x_te, y_te, train_i, test_i


def transform_input_to_numpy(spark):
    logger.info("Transforming prophet input data to numpy array ...")

    overall = time()

    logger.info("Reading reshaped data file ...")

    then = time()

    df = spark.read.csv(path=os.path.join(RMC.OUTPUT_DIR, RMC.PROPHET_INPUT_ALL_RESHAPED + '.csv'),
                        header=True, inferSchema=True)

    logger.info("Reading reshaped data file done in %s.", time_it(then, time()))

    logger.info("Collecting data ...")

    then = time()

    # SCENARIO's 2053 output is corrupted
    df = df.where('SCENARIO != 2053')

    df = df.orderBy('SCENARIO', 'MONTH')

    df = df.drop('SCENARIO', 'MONTH')

    col_nms = df.columns

    import pyspark.sql.functions as sf

    df = df.withColumn('FEATURES', sf.array(col_nms))

    df = df.drop(*col_nms)

    data = df.collect()

    logger.info("Collecting data done in %s.", time_it(then, time()))

    logger.info("Creating and saving numpy array ...")

    then = time()

    data = np.array(data)

    data = data.reshape(-1, 78)
    data = data.reshape(-1, 721, 78)

    np.save(os.path.join(RMC.OUTPUT_DIR, RMC.PROPHET_INPUT_ALL_NUMPY + '.npy'), data)

    logger.info("Creating and saving numpy array done in %s.", time_it(then, time()))

    logger.info("Transforming prophet input data to numpy array done in %s.", time_it(overall, time()))


def transform_output_to_numpy(spark):
    logger.info("Transforming prophet input data to numpy array ...")

    overall = time()

    logger.info("Reading output data file ...")

    then = time()

    df = spark.read.csv(path=os.path.join(RMC.OUTPUT_DIR, RMC.PROPHET_OUTPUT_ALL_PROPER + '.csv'),
                        header=True, inferSchema=True, sep=';')

    logger.info("Reading output data file done in %s.", time_it(then, time()))

    logger.info("Collecting data ...")

    then = time()

    df = df.orderBy('SCENARIO')
    df = df.drop('SCENARIO')

    data = df.collect()

    logger.info("Collecting data done in %s.", time_it(then, time()))

    logger.info("Creating and saving numpy array ...")

    then = time()

    data = np.array(data)

    np.save(os.path.join(RMC.OUTPUT_DIR, RMC.PROPHET_OUTPUT_ALL_NUMPY + '.npy'), data)

    logger.info("Creating and saving numpy array done in %s.", time_it(then, time()))

    logger.info("Transforming prophet ouput data to numpy array done in %s.", time_it(overall, time()))


def execute_proper():
    logger.info("Creating proper data files ...")

    then = time()

    wro = os.path.join(RMC.INPUT_DIR, RMC.PROPHET_INPUT_ALL + '.fac')
    pro = os.path.join(RMC.OUTPUT_DIR, RMC.PROPHET_INPUT_ALL_PROPER + '.csv')

    with open(wro, 'r') as orig:
        print("open orig")
        with open(pro, 'w') as copy:
            print("open copy")
            i = 0
            for line in orig:
                if i >= 2:
                    copy.write(line)

                i += 1

                if i % 100000 == 0:
                    logger.info("Creating proper input data file ... %3.2f%%", (i * 100.0 / 780081))

    wro = os.path.join(RMC.INPUT_DIR, RMC.PROPHET_OUTPUT_ALL + '.csv')
    pro = os.path.join(RMC.OUTPUT_DIR, RMC.PROPHET_OUTPUT_ALL_PROPER + '.csv')

    with open(wro, 'r') as orig:
        print("open orig")
        with open(pro, 'w') as copy:
            print("open copy")
            i = 0
            for line in orig:
                # SCENARIO's 2053 output is corrupted
                if i != 2053:
                    copy.write(line.replace(',', ''))
                else:
                    print(line)

                i += 1

                if i % 1000 == 0:
                    logger.info("Creating proper output data file ... %3.2f%%", (i * 100.0 / 10001))

    logger.info("Creating proper data files done in %s.", time_it(then, time()))


def execute_transpose(spark):
    logger.info("Transposing data ...")

    overall = time()

    logger.info("Reading proper data file ...")

    then = time()

    df = spark.read.csv(path=os.path.join(RMC.OUTPUT_DIR, RMC.PROPHET_INPUT_ALL_PROPER + '.csv'),
                        header=True, inferSchema=True)

    logger.info("Reading proper data file done in %s.", time_it(then, time()))

    logger.info("Collecting distinct value column names ...")

    then = time()

    import pyspark.sql.functions as sf

    df = df.withColumn('EC_CL_MS_OS', sf.concat(df.ECONOMY, sf.lit('_'), df.CLASS, sf.lit('_'),
                                                df.MEASURE, sf.lit('_'), df.OS_TERM))

    df = df.drop('!6', 'ECONOMY', 'CLASS', 'MEASURE', 'OS_TERM')

    # this is to provide faster test, needs to be uncommented later on
    #df = df.select('SCENARIO', 'EC_CL_MS_OS', '201712', '201801')

    # this is way cheaper to do now than to wait until all months have been transposed as rows
    val_col_nms = sorted(df.select('EC_CL_MS_OS').distinct().rdd.map(lambda row: row[0]).collect())

    logger.info("Collecting distinct value column names done in %s.", time_it(then, time()))

    logger.info("Transposing month columns as rows ...")

    then = time()

    keep_col_nms = ['SCENARIO', 'EC_CL_MS_OS']

    mo_col_nms = [c for c in df.columns if c not in keep_col_nms]

    mo_val_cols = sf.explode(sf.array(
        [sf.struct(sf.lit(c).alias("MONTH"), sf.col(c).alias("VALUE")) for c in mo_col_nms])).alias('MONTH_VALUE')

    df = df.select(keep_col_nms + [mo_val_cols]).select(keep_col_nms + ['MONTH_VALUE.MONTH', 'MONTH_VALUE.VALUE'])

    logger.info("Transposing month columns as rows done in %s.", time_it(then, time()))

    logger.info("Selecting with value columns ...")

    then = time()

    val_cols = [sf.when(sf.col('EC_CL_MS_OS') == c, sf.col('VALUE')).otherwise(None).alias(c)
                for c in val_col_nms]

    max_agg_cols = [sf.max(sf.col(c)).alias(c) for c in val_col_nms]

    df = df.select(sf.col('SCENARIO'), sf.col('MONTH'), *val_cols)

    logger.info("Selecting with value columns done in %s.", time_it(then, time()))

    logger.info("Aggregating value columns ...")

    then = time()

    df = df.groupBy('SCENARIO', 'MONTH').agg(*max_agg_cols)

    logger.info("Aggregating value columns done in %s.", time_it(then, time()))

    logger.info("Saving reshaped data to file ...")

    if os.path.exists(os.path.join(RMC.OUTPUT_DIR, 'tmp.csv')):
        rmtree(os.path.join(RMC.OUTPUT_DIR, 'tmp.csv'))

    df.write.csv(path=os.path.join(RMC.OUTPUT_DIR, 'tmp.csv'), mode='overwrite', header=True)

    if os.path.exists(os.path.join(RMC.OUTPUT_DIR, RMC.PROPHET_INPUT_ALL_RESHAPED + '.csv')):
        rmtree(os.path.join(RMC.OUTPUT_DIR, RMC.PROPHET_INPUT_ALL_RESHAPED + '.csv'))

    os.rename(src=os.path.join(RMC.OUTPUT_DIR, 'tmp.csv'),
                               dst=os.path.join(RMC.OUTPUT_DIR, RMC.PROPHET_INPUT_ALL_RESHAPED + '.csv'))

    logger.info("Saving reshaped data to file done in %s.", time_it(then, time()))

    logger.info("Transposing data done in %s.", time_it(overall, time()))

    return df


def execute_to_numpy(spark):
    logger.info("Saving data as numpy arrays ...")

    overall = time()

    transform_input_to_numpy(spark)

    transform_output_to_numpy(spark)

    logger.info("Saving data as numpy array done in %s.", time_it(overall, time()))


def execute_trim_and_split():
    logger.info("Starting initial data preparation ...")

    train_x, train_y, _, _, _, _, _, _, _ = load_all_data(
        train_set=True,
        val_set=False,
        test_set=False,
        init=True)

    print(train_x.shape, train_y.shape)
    logger.info("Trimming training data ...")

    train_x, train_y = trim_data(train_x, train_y, years=MLC.YEARS)
    print(train_x.shape, train_y.shape)

    logger.info("Trimming training data done.")

    logger.info("Splitting prepared training data ...")

    train_x_as_test = None
    train_y_as_test = None

    if MLC.TEST_SIZE == 0:
        train_x_as_test = train_x
        train_y_as_test = train_y

    train_x, train_y, val_x, val_y, test_x, test_y, train_i, val_i, test_i = split_data(x=train_x, y=train_y,
                                                                train_size=MLC.TRAIN_SIZE,
                                                                val_size=MLC.VAL_SIZE,
                                                                test_size=MLC.TEST_SIZE)

    if MLC.TEST_SIZE == 0:
        test_x = train_x_as_test
        test_y = train_y_as_test
        test_i = range(len(test_y))

    logger.info("Splitting prepared training data done.")

    save_all_data(
        train_x=train_x,
        train_y=train_y,
        train_i=train_i,
        val_x=val_x,
        val_y=val_y,
        val_i=val_i,
        test_x=test_x,
        test_y=test_y,
        test_i=test_i)


def main():
    overall = time()

    logger.info("Main script started ...")

    proper = False
    transpose = False
    to_numpy = False
    trim_and_split = False

    spark = None
    sc = None

    for arg in sys.argv[1:]:
        if arg == 'proper':
            proper = True
        if arg == 'transpose':
            transpose = True
        elif arg == 'to_numpy':
            to_numpy = True
        elif arg == 'trim_and_split':
            trim_and_split = True

    if not proper and not transpose and not to_numpy and not trim_and_split:
        proper = True
        transpose = True
        to_numpy = True
        trim_and_split = True

    if transpose or to_numpy:
        from pyspark import SparkContext
        from pyspark.sql import SparkSession

        sc = SparkContext("local[3]", "test")
        spark = SparkSession(sc)

    if proper:
        execute_proper()

    if transpose:
        execute_transpose(spark)

    if to_numpy:
        execute_to_numpy(spark)

    if trim_and_split:
        execute_trim_and_split()

    if sc is not None:
        sc.stop()

    logger.info("Main script done in %s.", time_it(overall, time()))


if __name__ == "__main__":
    main()