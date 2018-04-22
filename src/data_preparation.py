import sys
import os
import pickle
import logging.config
from time import time
from math import sqrt
from shutil import copyfile, rmtree

import numpy as np

import pandas as pd

from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as sf

from riskm_full import RMC, logger, time_it


def execute_proper_header():
    logger.info("Creating proper data file ...")

    then = time()

    fac = os.path.join(RMC.INPUT_DIR, RMC.PROPHET_INPUT_ALL + '.fac')
    csv = os.path.join(RMC.OUTPUT_DIR, RMC.PROPHET_INPUT_ALL_PROPER_HEADER + '.csv')

    with open(fac, 'r') as orig:
        with open(csv, 'w') as copy:
            i = 0
            for line in orig:
                if i >= 2:
                    copy.write(line)

                i += 1

                if i % 10000 == 0:
                    logger.info("Creating proper data file ... %3.2f%%", (i * 100.0 / 780081))

    logger.info("Creating proper data file done in %s.", time_it(then, time()))


def execute_transpose(spark):
    logger.info("Transposing data ...")

    overall = time()

    logger.info("Reading proper data file ...")

    then = time()

    df = spark.read.csv(path=os.path.join(RMC.OUTPUT_DIR, RMC.PROPHET_INPUT_ALL_PROPER_HEADER + '.csv'),
                        header=True, inferSchema=True)

    logger.info("Reading proper data file done in %s.", time_it(then, time()))

    logger.info("Collecting distinct value column names ...")

    then = time()

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
    logger.info("Saving data as numpy array ...")

    overall = time()

    logger.info("Reading reshaped data file ...")

    then = time()

    df = spark.read.csv(path=os.path.join(RMC.OUTPUT_DIR, RMC.PROPHET_INPUT_ALL_RESHAPED + '.csv'),
                        header=True, inferSchema=True)

    logger.info("Reading reshaped data file done in %s.", time_it(then, time()))

    logger.info("Collecting data ...")

    then = time()

    df = df.orderBy('SCENARIO', 'MONTH')

    df = df.drop('SCENARIO', 'MONTH')

    col_nms = df.columns

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

    logger.info("Saving data as numpy array done in %s.", time_it(overall, time()))


def main():
    overall = time()

    logger.info("Main script started ...")

    proper_header = False
    transpose = False
    to_numpy = False

    for arg in sys.argv[1:]:
        if arg == 'proper_header':
            proper_header = True
        if arg == 'transpose':
            transpose = True
        elif arg == 'to_numpy':
            to_numpy = True


    if not proper_header and not transpose and not to_numpy:
        proper_header = True
        transpose = True
        to_numpy = True

    sc = SparkContext("local[3]", "test")
    spark = SparkSession(sc)

    if proper_header:
        execute_proper_header()

    if transpose:
        execute_transpose(spark)

    if to_numpy:
        execute_to_numpy(spark)

    sc.stop()

    logger.info("Main script done in %s.", time_it(overall, time()))


if __name__ == "__main__":
    main()