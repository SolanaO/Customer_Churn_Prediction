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

################################################################
################################################################

## MODELs EVALUATORS FUNCTIONS

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
    OUTPUT:
        none - table of metrics is displayed

    """

    # extract
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
    print("auc_roc................%6.3f" % roc_auc)
    print("auc_pr.................%6.3f" % pr_auc)


# function to print the ROC and PR curves side by side
def plot_roc_pr_curves(predictions, model_name):

    """
    Calculates ROC-AUC and PR-AUC scores and plots the ROC and PR curves.
    Uses Pandas dataframes.

    INPUT:
        predictions (PySpark dataframe) - contains probability predictions, label column
        model_name (str) - classifier abbreviated name

    OUTPUT:
        none - two plots are displayed

    """

    # transform predictions PySpark dataframe into Pandas dataframe
    pred_pandas = predictions.select(predictions.label,   predictions.probability).toPandas()

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


# function to plot two roc and two pr curves on two side by side plots
def plot_roc_pr_curves(predictions_model1, predictions_model2, model1, model2):

    """
    Plots the ROC and PR curves for two models on the same graphs.

    INPUT:
        predictions_model1 (PySpark dataframe) - contains probability predictions for the first model
        predictions_model2 (PySpark dataframe) - contains probability predictions for the second model
        model1, model2 (str) - abreviated model names

    OUTPUT:
        none - two plots are displayed

    """

    # transform predictions PySpark dataframe into Pandas dataframe
    pred1_pandas = predictions_model1.select(predictions_model1.label,
                                             predictions_model1.probability).toPandas()
    pred2_pandas = predictions_model2.select(predictions_model2.label,
                                             predictions_model2.probability).toPandas()

    # calculate roc_auc scores
    roc_auc1 = roc_auc_score(pred1_pandas.label, pred1_pandas.probability.str[1])
    # calculate roc_auc scores
    roc_auc2 = roc_auc_score(pred2_pandas.label, pred2_pandas.probability.str[1])

    # calculate roc curves for model 1
    fpr1, tpr1, _ = roc_curve(pred1_pandas.label, pred1_pandas.probability.str[1])
    # calculate roc curves for model 2
    fpr2, tpr2, _ = roc_curve(pred2_pandas.label, pred2_pandas.probability.str[1])

    # calculate precision, recall for each threshold for the first model
    precision1, recall1, _ = precision_recall_curve(pred1_pandas.label, pred1_pandas.probability.str[1])
    # calculate pr auc score for the first model
    pr_auc1 = auc(recall1, precision1)

    # calculate precision, recall for each threshold for the second model
    precision2, recall2, _ = precision_recall_curve(pred2_pandas.label, pred2_pandas.probability.str[1])
    # calculate pr auc score for the second model
    pr_auc2 = auc(recall2, precision2)

    # create figure which contains two subplots
    plt.figure(figsize=[12,6])

    plt.subplot(121)

    # plot the roc curve for the model1
    plt.plot(ns_fpr1, ns_tpr1, linestyle='--', label='No Skill')
    plt.plot(fpr1, tpr1, marker='.', color='firebrick', label=model1 + ': ROC AUC = %.3f' % (roc_auc1))
    plt.plot(fpr2, tpr2, marker='.', color='green', label=model2 + ': ROC AUC = %.3f' % (roc_auc2))

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # figure title
    plt.title("ROC Curves")

    plt.subplot(122)

    # plot the precision-recall curves
    plt.plot(recall1, precision1, marker='.', color="firebrick", label=model1+ ': PR AUC = %.3f' % (pr_auc1))
    plt.plot(recall2, precision2, marker='.', color="green", label=model2+': PR AUC = %.3f' % (pr_auc2))

    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # figure title
    plt.title("Precision-Recall Curves")
