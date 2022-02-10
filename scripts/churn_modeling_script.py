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

# libraries and packages for modeling
from pyspark.ml import Pipeline

from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)
from pyspark.ml.feature import (
    OneHotEncoder,
    OneHotEncoderModel
)

from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier,
    MultilayerPerceptronClassifier,
    LinearSVC,
    NaiveBayes
)

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# build a Spark session using the SparkSession APIs
spark = (SparkSession
    .builder
    .appName("Sparkify")
    .getOrCreate())

spark.sparkContext.setLogLevel("ERROR")

# import python libraries
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import sklearn metrics related packages
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc, roc_curve, roc_auc_score


## BUILD PIPELINES

# split the features and the label
CAT_FEATURES = ["level"]
CONT_FEATURES = ["nr_songs", "nr_likes", "nr_dislikes", "nr_friends",
"nr_downgrades", "nr_upgrades", "nr_error", "nr_settings", "nr_ads",
"nr_sessions", "n_acts", "acts_per_session", "songs_per_session",
"ads_per_session", "init_days_interv", "tenure_days_interv", "active_days"]
CHURN_LABEL = "churn"

# create labels and features
PREDCOL="prediction"
LABELCOL="label"
FEATURESCOL = "features"

def build_full_pipeline(classifier):
    """
    Combines all the stages of the processing and modeling.
    """
    # stages in the pipeline
    stages = []

    # encode the labels
    label_indexer =  StringIndexer(inputCol=CHURN_LABEL, outputCol=LABELCOL)
    # add the indexer to the pipeline
    stages += [label_indexer]

    # encode the binary features
    bin_assembler = VectorAssembler(inputCols=CAT_FEATURES, outputCol="bin_features")
    # add the bin assembler to the pipeline
    stages += [bin_assembler]

    # encode the continuous features
    cont_assembler = VectorAssembler(inputCols = CONT_FEATURES, outputCol=FEATURESCOL)
    # add the vector assembler to the pipeline
    stages += [cont_assembler]
    # normalize the continuous features
    cont_scaler = StandardScaler(inputCol="cont_features", outputCol="cont_scaler", withStd=True , withMean=True)
    # add the scaler to the pipeline
    stages += [cont_scaler]

    # pass all to the vector assembler to create a single sparse vector
    all_assembler = VectorAssembler(inputCols=["bin_features", "cont_scaler"],  outputCol=FEATURESCOL)
    # add the vector assemble to the pipeline
    stages += [all_assembler]

    # add the model to the pipeline
    stages += [classifier]

    # create a pipeline
    pipeline = Pipeline(stages=stages)

    return pipeline

# implement K-fold cross validation and grid search
def grid_search_model(pipeline, param):
    """
    Creates a cross validation object and performs grid search
    over a set of parameters.

    INPUT:
        param = grid of parameters
        pipeline = model pipeline

    OUTPUT:
        cv = cross validation object
    """
    evaluator = BinaryClassificationEvaluator()
    cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=param,
                    evaluator=evaluator,
                    numFolds=5,
                    parallelism=2)
    return cv


## Pipeline for Stacking Model - Meta Classifier

# split the features and the label
META_FEATURES = ["pred_lr", "pred_rf","pred_gbt", "pred_mlpc", "pred_lsvc"]
META_CONT_FEATURES = ["prob_lr", "prob_rf", "prob_mlpc"]
META_LABEL_COL = "label"

# create labels and features
meta_predCol="meta_prediction"
meta_labelCol="meta_label"
meta_featuresCol = "meta_features"

def build_meta_pipeline(meta_classifier):
    """
    Combines all the stages of the meta features processing.
    """
    # stages in the pipeline
    stages = []

    # encode the labels
    label_indexer =  StringIndexer(inputCol=META_LABEL_COL, outputCol=meta_labelCol)
    stages += [label_indexer]

    # encode the binary features
    bin_assembler = VectorAssembler(inputCols=META_FEATURES, outputCol="bin_features")
    stages += [bin_assembler]

    # encode the continuous features
    cont_assembler = VectorAssembler(inputCols = META_CONT_FEATURES, outputCol="cont_features")
    stages += [cont_assembler]
    # normalize the continuous features
    cont_scaler = StandardScaler(inputCol="cont_features", outputCol="cont_scaler",  withStd=True , withMean=True)
    stages += [cont_scaler]

    # pass all to the vector assembler to create a single sparse vector
    all_assembler = VectorAssembler(inputCols=["bin_features", "cont_scaler"],
                                    outputCol=meta_featuresCol)
    stages += [all_assembler]

    # add the models to the pipeline
    stages += [meta_classifier]

    # create a pipeline
    pipeline = Pipeline(stages=stages)

    return pipeline
