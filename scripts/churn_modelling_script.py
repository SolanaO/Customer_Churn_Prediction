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


## LOAD DATA

def load_data(file_path):
    """
    Loads the raw dataset in Spark.

    INPUT:
            (str) - path for datafile
    OUTPUT:
            (PySpark dataframe) - dataframe of raw data

    """

    print("Loading the dataset ...")
    df = spark.read.json(file_path)
    print("Dataset is loaded...")

    return df

# path for the train set file
path_trainset = "data/mini_train.json"
# upload the train data
df_train = load_data(path_trainset)
# path for the train set file
path_testset = "data/mini_test.json"
# upload the train data
df_test = load_data(path_testset)

# toggle the memory
train_cached = df_train.cache()
test_cached = df_test.cache()

## MODEL EVALUATORS

# function to compute relevant metrics for binary classification
def conf_metrics(dataset):

    """
    Calculates the metrics associated to the confusion matrix.

    INPUT:
        dataset (pyspark.sql.DataFrame) - a dataset that contains
                            labels and predictions
    OUTPUT:
        accuracy (float) - metric
        precision (float) - metric
        recall (float) - metric
        F1 (float) - metric
    """


    # calculate the elements of the confusion matrix
    tn = dataset.where((dataset[labelCol]==0) & (dataset[predCol]==0)).count()
    tp = dataset.where((dataset[labelCol]==1) & (dataset[predCol]==1)).count()
    fn = dataset.where((dataset[labelCol]==1) & (dataset[predCol]==0)).count()
    fp = dataset.where((dataset[labelCol]==0) & (dataset[predCol]==1)).count()

    # calculate accuracy, precision, recall, and F1-score
    accuracy = (tn + tp) / (tn + tp + fn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 =  2 * (precision*recall) / (precision + recall)

    return accuracy, precision, recall, f1


# function to display the metrics of interest
def display_metrics(dataset, roc_cl, pr_cl):

    """
    Prints evaluation metrics for the model.

    INPUT:
         dataset (pyspark.sql.DataFrame) - a dataset that contains
                                labels and predictions

    """

    accuracy = conf_metrics(dataset)[0]
    precision = conf_metrics(dataset)[1]
    recall = conf_metrics(dataset)[2]
    f1 = conf_metrics(dataset)[3]

    print("")
    print("Confusion Matrix")
    dataset.groupBy(dataset[labelCol], dataset[predCol]).count().show()
    print("")
    print("accuracy...............%6.3f" % accuracy)
    print("precision..............%6.3f" % precision)
    print("recall.................%6.3f" % recall)
    print("F1.....................%6.3f" % f1)
    print("auc_roc................%6.3f" % roc_cl)
    print("auc_pr.................%6.3f" % pr_cl)


# function to print the ROC and PR curves
def plot_roc_pr_curves(predictions, model_name):

    """
    Calculates ROC-AUC and PR-AUC scores and plots the ROC and PR curves.

    INPUT:
        predictions (PySpark dataframe) - contains probability predictions, label column
        model_name (str) - classifier name

    OUTPUT:
        none - two plots are displayed

    """

    # transform predictions PySpark dataframe into Pandas dataframe
    pred_pandas = predictions.select(predictions.label, predictions.probability).toPandas()

    # calculate roc_auc score
    roc_auc = roc_auc_score(pred_pandas.label, pred_pandas.probability.str[1])
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(pred_pandas.label))]
    # calculate roc curves
    fpr, tpr, _ = roc_curve(pred_pandas.label, pred_pandas.probability.str[1])
    ns_fpr, ns_tpr, _ = roc_curve(pred_pandas.label, ns_probs)

    # calculate precision, recall for each threshold
    precision, recall, _ = precision_recall_curve(pred_pandas.label, pred_pandas.probability.str[1])
    # calculate pr auc score
    pr_auc = auc(recall, precision)


    # create figure which contains two subplots
    plt.figure(figsize=[12,6])

    plt.subplot(121)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', color='firebrick', label='ROC AUC = %.3f' % (roc_auc))

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # figure title
    plt.title("ROC Curve:" + model_name)

    plt.subplot(122)

    # plot the precision-recall curves

    ns_line = len(pred_pandas[pred_pandas.label==1]) / len(pred_pandas.label)
    plt.plot([0, 1], [ns_line, ns_line], linestyle='--', label='No Skill')
    plt.plot(recall, precision, marker='.', color='firebrick', label='PR AUC = %.3f' % (pr_auc))

    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # figure title
    plt.title("Precision-Recall Curve:" + model_name)

    # show the plot
    plt.show()


## BUILD PIPELINES

# split the features and the label
CAT_FEATURES = ["level"]
CONT_FEATURES = ["nr_songs", "nr_likes", "nr_dislikes", "nr_friends",   "nr_downgrades", "nr_upgrades", "nr_error", "nr_settings", "nr_ads", "nr_sessions", "n_acts", "acts_per_session", "songs_per_session", "ads_per_session", "init_days_interv", "tenure_days_interv", "active_days"]
CHURN_LABEL = "churn"

def build_full_pipeline(classifier):
    """
    Combines all the stages of the processing and modeling.
    """
    # stages in the pipeline
    stages = []

    # encode the labels
    label_indexer =  StringIndexer(inputCol=CHURN_LABEL, outputCol="label")
    stages += [label_indexer]

    # encode the binary features
    bin_assembler = VectorAssembler(inputCols=CAT_FEATURES,  outputCol="bin_features")
    stages += [bin_assembler]

    # encode the continuous features
    cont_assembler = VectorAssembler(inputCols = CONT_FEATURES, outputCol="cont_features")
    stages += [cont_assembler]
    # normalize the continuous features
    cont_scaler = StandardScaler(inputCol="cont_features", outputCol="cont_scaler",
                                 withStd=True , withMean=True)
    stages += [cont_scaler]

    # pass all to the vector assembler to create a single sparse vector
    all_assembler = VectorAssembler(inputCols=["bin_features", "cont_scaler"],  outputCol="features")
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
