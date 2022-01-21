# import PySpark libraries and packages

import pyspark
from pyspark import SparkContext
from pyspark import SparkConf

from pyspark.sql import SparkSession
from pyspark.sql.window import Window as W

from pyspark.sql.types import (
    StringType,
    IntegerType,
    DateType,
    TimestampType,
    )

from pyspark.sql.functions import (
    min as Fmin, max as Fmax,
    sum as Fsum, round as Fround,

    col, lit,
    first, last,
    desc, asc,
    avg, count, countDistinct,
    when, isnull, isnan,
    from_unixtime,
    datediff,
    )

# build a Spark session using the SparkSession APIs

spark = (SparkSession
        .builder
        .appName("Sparkify")
        .getOrCreate())

spark.sparkContext.setLogLevel("ERROR")

# import python libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# import library for enhanced plotting

import seaborn as sns
sns.set_style("darkgrid")
colors = sns.color_palette('PuBuGn_r')

###############################################
###############################################

def load_data(file_path):
    """
    Loads the raw dataset in PySpark.

    INPUT:
            (str) - path for datafile
    OUTPUT:
            (PySpark dataframe) - dataframe of raw data

    """

    print("Loading the dataset ...")
    df = spark.read.json(file_path)
    print("Dataset is loaded...")

    return df


def clean_data(df):
    """
    Performs basic cleaning operations on the raw data:
        - removes entries with missing userId
        - rescale timestamp columns to seconds
        - drop unnecesary columns
            - personal information columns
            - song information columns
            - web and browser information
            - timestamp columns in miliseconds

    INPUT:
        (PySpark dataframe) - dataframe of raw data
    OUTPUT:
        (PySpark dataframe) - dataframe of cleaned data
    """

    # print message to indicate the start of the process
    print("Cleaning the data ...")

    # print a count of rows before cleaning
    initial_records = df.count()
    print("Dataset has {} rows initially.".format(initial_records))

    # filter out all the records without userId
    df = df.where(df.userId != "")

    # rescale the timestamp to seconds (initially in miliseconds)
    df = df.withColumn("log_ts", df.ts/1000.0)
    df = df.withColumn("reg_ts", df.registration/1000.0)

    # drop several unnecessary columns
    cols_to_drop = ("firstName", "lastName", "location",
                    "artist", "song", "length",
                    "userAgent", "method", "status",
                    "ts", "registration"
                   )
    df = df.drop(*cols_to_drop)


    # print end of process message
    print("Finished cleaning the data ...")

    # print a count of rows after cleaning
    removed_rows = initial_records - df.count()

    print("Cleaned dataset has {} rows, {} rows were removed". \
        format(df.count(), initial_records - df.count()))

    return df


def save_data(df, data_path):
    """
    Saves the PySpark dataframe to a file.

    INPUT:
            df (PySpark dataframe) - data to be saved
                    data_path (str) - path for datafile
    OUTPUT:
            none

    """

    df.write.json(data_path)



def preprocess_data(df):

    """
    Prepare the data for modeling via creating several features.

        - reg_date (date) - month-year of the registration

        - create windows grouped on userId and sessionId

         - firstevent_ts (timestamp) - first time an user is active
         - lastevent_ts (timestamp) - last time an user is active

         - init_days_interv (float) - days between registration and first activity
         - tenure_days_interv (float) - days between registration and last activity
         - active_days (float) - days the user has some activity on the platform
         - session_h (float) - session's duration in hours

     INPUT:
         df (PySpark dataframe) - cleaned dataframe
     OUTPUT:
         df (PySpark dataframe) - dataframe with the listed features added
    """

    # extract registration month and year from timestamp
    df = df.withColumn("reg_date", from_unixtime(col("reg_ts"), "MM-yyyy"))

    # create window: data grouped by userId, time ordered
    win_user = (W.partitionBy("userId")
            .orderBy("log_ts")
            .rangeBetween(W.unboundedPreceding, W.unboundedFollowing))

    # create window: data grouped by sessionId and userId, time ordered
    win_user_session = (W.partitionBy("sessionId", "userId")
                        .orderBy("log_ts")
                        .rangeBetween(W.unboundedPreceding, W.unboundedFollowing))

    # record the first time an user is active
    df = df.withColumn("firstevent_ts", first(col("log_ts")).over(win_user))
    # record the last time an user is active
    df = df.withColumn("lastevent_ts", last(col("log_ts")).over(win_user))

    # warmup time = registration time to first event in days
    df = df.withColumn("init_days_interv",
                       (col("firstevent_ts").cast("long")-col("reg_ts").cast("long"))/(24*3600))

    # tenure time = registration time to last event in days
    df = df.withColumn("tenure_days_interv",
                       (col("lastevent_ts").cast("long")-col("reg_ts").cast("long"))/(24*3600))

    # active time =  days between the first event and the last event in days
    df = df.withColumn("active_days",
                       (col("lastevent_ts").cast("long")-col("firstevent_ts").cast("long"))/(24*3600))

    # create column that records the individual session's duration in hours
    df = df.withColumn("session_h",
                    (last(df.log_ts).over(win_user_session) \
                     - first(df.log_ts).over(win_user_session))/3600)

    # drop columns
    df = df.drop("reg_ts", "log_ts")

    return df


def build_features(df):

    """
    Features engineered to be used in modelling.

        - nr_songs (int) - total number of songs user listened to
        - nr_playlist (int) - number of songs added to the playlist

        - nr_friends (int) - number of friends added through "Add Friend"

        - nr_likes (int) - total number of "Thumbs Up" of the user
        - nr_dislikes (int) - total number of "Thumbs Down" of the user

        - nr_downgrades (int) - total number of visits to "Downgrade" page by the user
        - nr_upgrades (int) - total number of visits to "Upgrade" page by the user

        - nr_home (int) - total number of visits to "Home" page by the user
        - nr_settings (int) - total number of visits to "Settings" page by the user

        - nr_error (int) - total number of errors encountered by the user

        - nr_ads (int) - total number of ads the user got
        - nr_sessions (int) - number of sessions of the user
        - n_acts (int) - total number of actions taken by the user

        - avg_sess_h (float) - average session length in hours
        - acts_per_session (float) - average number of actions per session for the user
        - songs_per_session (float) - average numer of songs listened per session by the user
        - ads_per_session (float) - average number of ads per session, received by user

        - init_days_interv (int) - time interval in days from registration to the first action of the user
        - tenure_days_interv (int) - time interval in days from registration to the last action of the user
        - active_days (int) - number of days the user was active on the platform

        - gender (binary) - 1 for F (female), 0 for M (male)
        - level (binary) - 1 for paid, 0 for free

        - churn (binary) - 1 for "Cancellation Confirmation" page visit, 0 otherwise

    INPUT:
        df (PySpark dataframe) - preprocessed dataframe
    OUTPUT:
        df_feats (PySpark dataframe) - dataframe that contains engineered features
    """

    df_feats = df.groupBy("userId") \
        .agg(

            # count user's individual actions using all page visits

            count(when(col("page") == "NextSong", True)).alias("nr_songs"),
            count(when(col("page") == "Add to Playlist", True)).alias("nr_playlist"),

            count(when(col("page") == "Add Friend", True)).alias("nr_friends"),

            count(when(col("page") == "Thumbs Up", True)).alias("nr_likes"),                count(when(col("page") == "Thumbs Down", True)).alias("nr_dislikes"),

            count(when(col("page") == "Downgrade", True)).alias("nr_downgrades"),
            count(when(col("page") == "Upgrade", True)).alias("nr_upgrades"),

            count(when(col("page") == "Home", True)).alias("nr_home"),
            count(when(col("page") == "Settings", True)).alias("nr_settings"),

            count(when(col("page") == "Error", True)).alias("nr_error"),

            count(when(col("page") == "Roll Advert", True)).alias("nr_ads"),

            # compute the number of sessions a user is in
            countDistinct("sessionId").alias("nr_sessions"),

            # find the total number of actions a user took
            countDistinct("itemInSession").alias("n_acts"),

            # compute the average session length in hours
            avg(col("session_h")).alias("avg_sess_h"),

            # compute the average number of page actions per sesssion - i.e. items in session
            (countDistinct("itemInSession")/countDistinct("sessionId")).alias("acts_per_session"),

            # compute the average number of songs per session
            (count(when(col("page") == "NextSong",                            True))/countDistinct("sessionId")).alias("songs_per_session"),

            # compute the average number of ads per session
             (count(when(col("page") == "Roll Advert",
                                          True))/countDistinct("sessionId")).alias("ads_per_session"),

            # days between registration and first activity
            first(col("init_days_interv")).alias("init_days_interv"),
            # the tenure time on the platform: from registration to last event in days
            first(col("tenure_days_interv")).alias("tenure_days_interv"),
            # number of days user visited the platform, is active on the platform
            first(col("active_days")).alias("active_days"),

            # encode the gender 1 for F and 0 for M
            first(when(col("gender") == "F", 1).otherwise(0)).alias("gender"),

            # encode the level (paid/free) according to the last record
            last(when(col("level") == "paid", 1).otherwise(0)).alias("level"),

            # flag those users that downgraded
            #last(when(col("page") == "Downgrade", 1).otherwise(0)).alias("downgrade"),

            # create the churn column that records if the user cancelled
            last(when(col("page") == "Cancellation Confirmation", 1).otherwise(0)).alias("churn"),
            )

    # columns to drop
    drop_cols = ("userId", "gender", "avg_sess_h",
                 "nr_playlist", "nr_home")
    # drop the columns
    #df_feats = df_feats.drop("userId")
    df_feats = df_feats.drop(*drop_cols)

    # drop the null values
    df_feats=df_feats.na.drop()

    return df_feats


SPLIT_VALS = [.7, .3]

# split the data into train and test sets

def split_data (df):

    """
    Split the dataset into training set and test set.
    Use a stratified sampling method.

    INPUT:
        df (PySpark dataframe) - dataframe
    OUTPUT:
        train_set, test_set (PySpark dataframes) - percentage split based on the provided values
    """

    # split dataframes between 0s and 1s
    zeros = df.filter(df["churn"]==0)
    ones = df.filter(df["churn"]==1)

    # split dataframes into training and testing
    train0, test0 = zeros.randomSplit(SPLIT_VALS, seed=1234)
    train1, test1 = ones.randomSplit(SPLIT_VALS, seed=1234)

    # stack datasets back together
    train_set = train0.union(train1)
    test_set = test0.union(test1)

    return train_set, test_set


def main():
    if len(sys.argv) == 4:
        dataset_filepath, train_filepath, test_filepath = sys.argv[1:]

        df = load_data(dataset_filepath)

        df_clean = clean_data(df)

        print('Preprocess data...')
        df_proc = preprocess_data(df_clean)

        print('Create features dataset...')
        df_feats = build_features(df_proc)

        SPLIT_VALS = [.7, .3]

        print('Split the features dataset...')
        train_set, test_set = split_data(df_feats)

        print('Saving train and test sets...')
        save_data(train_set, train_filepath)
        save_data(test_set, test_filepath)

        print('The train and test sets are saved!')

    else:
        print('Please provide the filepath for the Sparkify dataset ' \
              'as the first argument and the filepaths for the cleaned ' \
              'train and test datasets to be saved to as the second  and ' \
              'third arguments. \n\nExample: python ' \
              'churn_prepdata_script.py ../data/mini_sparkify_event_data.json  ../data/mini_sparkify_train_data.json  ../data/mini_sparkify_test_data.json')


if __name__ == '__main__':
    main()
