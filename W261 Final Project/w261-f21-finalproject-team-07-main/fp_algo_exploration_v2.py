# Databricks notebook source
# MAGIC %md
# MAGIC # Algorithm Exploration

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook implements Logistic Regression, Random Forest, SVM and Gradient Boosting Tree to our delayed flights dataset.
# MAGIC 
# MAGIC <a href='$./fp_main_notebook_final'>To return to main notebook click here</a>.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set-up blob storage

# COMMAND ----------

# init script to create the blob URL
blob_container = 'team07'
storage_account = 'team07'
secret_scope = 'team07'
secret_key = 'team07'
blob_url = f'wasbs://{blob_container}@{storage_account}.blob.core.windows.net'

# generates the SAS token
spark.conf.set(
  f'fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net',
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run imports

# COMMAND ----------

# imports
import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
from itertools import chain
from pyspark.sql import Row, Column
from pyspark.sql.types import BooleanType
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.feature import StandardScaler, Imputer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import the joined data

# COMMAND ----------

# MAGIC %md
# MAGIC We have a check-point at the blob storage with the data ready for modelling.

# COMMAND ----------

# read joined dataset
rawDataDF = spark.read.parquet(f'{blob_url}/joined_data_all_v1')

# print out number of rows
print(str(rawDataDF.count()) + ' rows in the data.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filter columns

# COMMAND ----------

# MAGIC %md
# MAGIC Filter only columns of interest.

# COMMAND ----------

# label and features of interest
cols = ['DEP_DEL15', 'YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'OP_CARRIER', 'ORIGIN', 'DEST', 'rolling_average', 'delay_2hrs_originport', 'delay_4hrs_originport', 'delay_8hrs_originport', 'delay_12hrs_originport', 'delay_2hrs_destport', 'delay_4hrs_destport', 'delay_8hrs_destport', 'delay_12hrs_destport', 'delay_2hrs_orgairline', 'delay_4hrs_orgairline', 'delay_8hrs_orgairline', 'delay_12hrs_orgairline', 'arrdelay_2hrs_originport', 'arrdelay_4hrs_originport', 'arrdelay_8hrs_originport', 'arrdelay_12hrs_originport', 'delay_2hrs_originhub', 'delay_4hrs_originhub', 'delay_8hrs_originhub', 'delay_12hrs_originhub', 'DEP_HOUR', 'Part_of_Day', 'WND_direction_angle', 'WND_speed', 'CIG_ceiling_height_dimension', 'VIS_distance_dimension', 'TMP_air_temperature', 'DEW_dew_point_temperature', 'SLP_sea_level_pressure', 'Bad_Weather_Prediction']

# filter cols of interest
filteredDataDF = rawDataDF.select(cols).cache()

# print out number of features
# minus two to account for the response and the year variables (which will only be used for splitting)
print(str(len(filteredDataDF.columns)-2) + ' features in the data.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split train and test data

# COMMAND ----------

# set 2019 as the hold-out test set
hold_out_variable = 'YEAR'
hold_out_threshold = '2019'

# split train and test sets
trainDF = filteredDataDF.filter(filteredDataDF[hold_out_variable]!=hold_out_threshold).cache()
testDF = filteredDataDF.filter(filteredDataDF[hold_out_variable]==hold_out_threshold).cache()

# print count of rows
train_years = sorted([x[0] for x in trainDF.select(hold_out_variable).distinct().collect()])
test_years = sorted([x[0] for x in testDF.select(hold_out_variable).distinct().collect()])
print(f'{trainDF.count()} rows in the train data, representing {hold_out_variable.lower()}s: {str(train_years)[1:-1]}.')
print(f'{testDF.count()} rows in the test data, representing {hold_out_variable.lower()}s: {str(test_years)[1:-1]}.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Oversample minority class

# COMMAND ----------

# MAGIC %md
# MAGIC Given our data is unbalanced (most of flights do not delay) we will do random oversampling in the minority class aiming to get to a 50-50% class balance in the training dataset.

# COMMAND ----------

# split the data given labels
minor_df = trainDF.filter(col('DEP_DEL15')==1).cache()
major_df = trainDF.filter(col('DEP_DEL15')==0).cache()

# compute the ratio between on-time and delayed flights
n_ontime = major_df.count()
n_delays = minor_df.count()
ratio = n_ontime/n_delays
print('The ratio of on-time to delayed flights is of {:0.2f}:1'.format(ratio))

# oversample the delayed flights
oversample_df = minor_df.sample(withReplacement=True, fraction=ratio, seed=123)
augmentedTrainDF = major_df.unionAll(oversample_df).cache()
augmentedTrainDF.groupBy('DEP_DEL15').count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split CV folds in the train data

# COMMAND ----------

# MAGIC %md
# MAGIC Creates a `foldCol` which specifies how we want to fold the data for cross-validation.

# COMMAND ----------

# extract number of distinct YEARS in the training data and create a map
fold_variable = 'YEAR'
fold_list = augmentedTrainDF.select(fold_variable).distinct().toPandas()[fold_variable]
mapping = {x: x - fold_list.min() for x in fold_list}
print(f'{fold_variable.capitalize()}, fold_number mapping: {mapping}')

# define number of Folds as nYEARS - 1
nFolds = len(fold_list) - 1
print(f'Total number of folds: {nFolds}')

# add 'foldCol' to TrainDF
mapping_expr = create_map([lit(x) for x in chain(*mapping.items())])
foldedTrainDF = augmentedTrainDF.withColumn('foldCol', mapping_expr[col(fold_variable)]).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Customize CrossValidator creating TimeSeriesCrossValidator

# COMMAND ----------

# MAGIC %md
# MAGIC Pyspark ML supports model selection through k-fold Cross Validation, but has not built-in methods for Time Series Cross Validation. In order to take benefit of a single Pyspark Pipeline, we decided to customize the CrossValidator class in order to change it behavior and made it supportive of Time Series Cross Validation. The method we will be using is Cross Validation on a Rolling Basis. We will define our training data splits by year. We will then use 2015 to train, and predict 2016. Then we will use 2015 and 2016 to train, and predict 2017. Lastly we will use 2015, 2016 and 2017 to train, and predict 2018. We are going to have a total of 3 kFolds / models. Performance metrics will be averaged for the 3 models for model selection purposes. Pyspark ML already allows the user to specify how it wants fold the data (user-specified fold numbers vs. random split) by using the `foldCol` argument, but the standard behavior still trains the model in all other folds except the fold selected for validation. In order to change this behavior, we will need to do an small change in the method `_kFold`. We will specify that the training folds must always come before the validation fold (never after). All the other functionalities of the class will be kept intact to ensure a smooth integration with the rest of the Pyspark Pipeline.

# COMMAND ----------

class TimeSeriesCrossValidator(CrossValidator):
    '''
    Customizes CrossValidator to perform time series cross validation on a rolling basis.
    User needs to provide `foldCol` with the fold numbers defined in a time ascending order
    (e.g. 2015 is assigned as fold 0, 2016 as fold 1, and so on).
    '''
    def _kFold(self, dataset):
        nFolds = self.getOrDefault(self.numFolds)
        foldCol = self.getOrDefault(self.foldCol)

        datasets = []
        if not foldCol:
            # Do random k-fold split.
            seed = self.getOrDefault(self.seed)
            h = 1.0 / nFolds
            randCol = self.uid + "_rand"
            df = dataset.select("*", rand(seed).alias(randCol))
            for i in range(nFolds):
                validateLB = i * h
                validateUB = (i + 1) * h
                condition = (df[randCol] >= validateLB) & (df[randCol] < validateUB)
                validation = df.filter(condition)
                train = df.filter(~condition)
                datasets.append((train, validation))
        else:
            # Use user-specified fold numbers.
            def checker(foldNum):
                if foldNum < 0 or foldNum > nFolds:
                    raise ValueError(
                        "Fold number must be in range [0, %s], but got %s." % (nFolds, foldNum)
                    )
                return True

            checker_udf = UserDefinedFunction(checker, BooleanType())
            for i in range(nFolds):
                training = dataset.filter(checker_udf(dataset[foldCol]) & (col(foldCol) <= lit(i))) # Training set always in the past
                validation = dataset.filter(
                    checker_udf(dataset[foldCol]) & (col(foldCol) == lit(i+1)) # Validation set always in the future
                )
                if training.rdd.getNumPartitions() == 0 or len(training.take(1)) == 0:
                    raise ValueError("The training data at fold %s is empty." % i)
                if validation.rdd.getNumPartitions() == 0 or len(validation.take(1)) == 0:
                    raise ValueError("The validation data at fold %s is empty." % i)
                datasets.append((training, validation))

        return datasets

# COMMAND ----------

# MAGIC %md
# MAGIC # Logistic Regression

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set the pipeline

# COMMAND ----------

# define categorical and continuous variables
categoricals = ['QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'DEP_HOUR', 'OP_CARRIER', 'ORIGIN', 'DEST', 'Bad_Weather_Prediction']
numerics = ['rolling_average', 'delay_2hrs_originport', 'delay_4hrs_originport', 'delay_8hrs_originport', 'delay_12hrs_originport', 'delay_2hrs_destport', 'delay_4hrs_destport', 'delay_8hrs_destport', 'delay_12hrs_destport', 'delay_2hrs_orgairline', 'delay_4hrs_orgairline', 'delay_8hrs_orgairline', 'delay_12hrs_orgairline', 'arrdelay_2hrs_originport', 'arrdelay_4hrs_originport', 'arrdelay_8hrs_originport', 'arrdelay_12hrs_originport', 'delay_2hrs_originhub', 'delay_4hrs_originhub', 'delay_8hrs_originhub', 'delay_12hrs_originhub', 'WND_direction_angle', 'WND_speed', 'CIG_ceiling_height_dimension', 'VIS_distance_dimension', 'SLP_sea_level_pressure', 'TMP_air_temperature', 'DEW_dew_point_temperature']

# define feature transformations
indexer = map(lambda c: StringIndexer(inputCol=c, outputCol=c+'_idx', handleInvalid = 'keep'), categoricals)
ohes = map(lambda c: OneHotEncoder(inputCol=c+'_idx', outputCol=c+'_class'), categoricals)
imputer = Imputer(strategy='median', inputCols = numerics, outputCols = numerics)
feature_cols = list(map(lambda c: c+'_idx', categoricals)) + numerics
vassembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures',
                        withStd=True, withMean=True)

# create a map of {idx: count of distinct values} for the categorial features
cat_features_map = {idx: trainDF.select(categoricals[idx]).distinct().count() for idx in range(len(categoricals))}
max_dist_values = sorted(cat_features_map.values())[-1]+1

# set-up the algorithm
lr = LogisticRegression(featuresCol='scaledFeatures', labelCol='DEP_DEL15', maxIter=100, regParam=0.1, elasticNetParam=0.5, 
                        standardization=False, family='binomial')
# assemble the pipeline
lr_transf_stages = list(indexer) + list(ohes) + [imputer] + [vassembler] + [scaler] + [lr]
lr_pipeline = Pipeline(stages=lr_transf_stages)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select best model using TimeSeriesCrossValidator

# COMMAND ----------

# build the parameter grid for model tuning
lr_paramGrid = ParamGridBuilder() \
              .addGrid(lr.regParam, [0.01, 0.1]) \
              .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
              .build()

# execute TimeSeriesCrossValidator for model tuning
lr_crossval = TimeSeriesCrossValidator(estimator=lr_pipeline,
                          estimatorParamMaps=lr_paramGrid,
                          evaluator=BinaryClassificationEvaluator(labelCol='DEP_DEL15', 
                                                                  metricName='areaUnderROC'),
                          parallelism=3,
                          foldCol='foldCol',
                          numFolds=nFolds)

# train the tuned model and establish our best model
start = time.time()
lr_cvModel = lr_crossval.fit(foldedTrainDF)
lr_model = lr_cvModel.bestModel
print(f'CV-training time: {time.time() - start} seconds')
print('')

# print best model params
print(f'Best Param (maxIter): {lr_model.stages[-1].getMaxIter()}')
print(f'Best Param (regParam): {lr_model.stages[-1].getRegParam()}')
print(f'Best Param (elasticNetParam): {lr_model.stages[-1].getElasticNetParam()}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evalute results in the training set

# COMMAND ----------

# evaluate results in the training set
predictions = lr_model.transform(trainDF.filter(trainDF.DEP_DEL15.isNotNull()))
eval_ = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='DEP_DEL15')

# store performance metrics in a dictionary
metrics = ['accuracy', 'weightedPrecision', 'weightedRecall', 'weightedFMeasure', 'precisionByLabel', 'recallByLabel', 'fMeasureByLabel']
results = {}
for m in metrics:
  if m in ['precisionByLabel', 'recallByLabel', 'fMeasureByLabel']:
    results[m] = [eval_.evaluate(predictions, {eval_.metricName: m, eval_.metricLabel:0.0}), 
                  eval_.evaluate(predictions, {eval_.metricName: m, eval_.metricLabel:1.0})]
  else:
    results[m] = eval_.evaluate(predictions, {eval_.metricName: m})

# print results
print('Performance metrics - TRAINING SET')
print('------------------------------------------------------------------------------------------------')
for x in results:
  print(f'{x}: {results[x]}')
  
# save results
lr_train = results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate results in the test set

# COMMAND ----------

# evaluate results in the test set
predictions = lr_model.transform(testDF)
eval_ = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='DEP_DEL15')

# store performance metrics in a dictionary
metrics = ['accuracy', 'weightedPrecision', 'weightedRecall', 'weightedFMeasure', 'precisionByLabel', 'recallByLabel', 'fMeasureByLabel']
results = {}
for m in metrics:
  if m in ['precisionByLabel', 'recallByLabel', 'fMeasureByLabel']:
    results[m] = [eval_.evaluate(predictions, {eval_.metricName: m, eval_.metricLabel:0.0}), 
                  eval_.evaluate(predictions, {eval_.metricName: m, eval_.metricLabel:1.0})]
  else:
    results[m] = eval_.evaluate(predictions, {eval_.metricName: m})

# print results
print('Performance metrics - TEST SET')
print('------------------------------------------------------------------------------------------------')
for x in results:
  print(f'{x}: {results[x]}')
  
# save results
lr_test = results

# COMMAND ----------

# MAGIC %md
# MAGIC # Random Forest Classfier

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set the pipeline

# COMMAND ----------

# define categorical and continuous variables
categoricals = ['QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'DEP_HOUR', 'OP_CARRIER', 'ORIGIN', 'DEST', 'Bad_Weather_Prediction']
numerics = ['rolling_average', 'delay_2hrs_originport', 'delay_4hrs_originport', 'delay_8hrs_originport', 'delay_12hrs_originport', 'delay_2hrs_destport', 'delay_4hrs_destport', 'delay_8hrs_destport', 'delay_12hrs_destport', 'delay_2hrs_orgairline', 'delay_4hrs_orgairline', 'delay_8hrs_orgairline', 'delay_12hrs_orgairline', 'arrdelay_2hrs_originport', 'arrdelay_4hrs_originport', 'arrdelay_8hrs_originport', 'arrdelay_12hrs_originport', 'delay_2hrs_originhub', 'delay_4hrs_originhub', 'delay_8hrs_originhub', 'delay_12hrs_originhub', 'WND_direction_angle', 'WND_speed', 'CIG_ceiling_height_dimension', 'VIS_distance_dimension', 'SLP_sea_level_pressure', 'TMP_air_temperature', 'DEW_dew_point_temperature']

# define feature transformations
indexer = map(lambda c: StringIndexer(inputCol=c, outputCol=c+'_idx', handleInvalid = 'keep'), categoricals)
ohes = map(lambda c: OneHotEncoder(inputCol=c+'_idx', outputCol=c+'_class'), categoricals)
imputer = Imputer(strategy='median', inputCols = numerics, outputCols = numerics)
feature_cols = list(map(lambda c: c+'_idx', categoricals)) + numerics
vassembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures',
                        withStd=True, withMean=True)

# create a map of {idx: count of distinct values} for the categorial features
cat_features_map = {idx: trainDF.select(categoricals[idx]).distinct().count() for idx in range(len(categoricals))}
max_dist_values = sorted(cat_features_map.values())[-1]+1

# set-up the algorithm
rfw = RandomForestClassifier(featuresCol='features', labelCol='DEP_DEL15', numTrees = 500, featureSubsetStrategy='sqrt', 
                             maxDepth=6, maxBins=max_dist_values, impurity='gini', seed=123)
# assemble the pipeline
rfw_transf_stages = list(indexer) + [imputer] + [vassembler] + [rfw]
rfw_pipeline = Pipeline(stages=rfw_transf_stages)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select best model using TimeSeriesCrossValidator

# COMMAND ----------

# build the parameter grid for model tuning
rfw_paramGrid = ParamGridBuilder() \
               .addGrid(rfw.numTrees, [100, 500]) \
               .addGrid(rfw.maxDepth, [4, 8]) \
               .build()

# execute TimeSeriesCrossValidator for model tuning
rfw_crossval = TimeSeriesCrossValidator(estimator=rfw_pipeline,
                          estimatorParamMaps=rfw_paramGrid,
                          evaluator=BinaryClassificationEvaluator(labelCol='DEP_DEL15', 
                                                                  metricName='areaUnderROC'),
                          parallelism=3,
                          foldCol='foldCol',
                          numFolds=nFolds)

# train the tuned model and establish our best model
start = time.time()
rfw_cvModel = rfw_crossval.fit(foldedTrainDF)
rfw_model = rfw_cvModel.bestModel
print(f'CV-training time: {time.time() - start} seconds')
print('')

# print best model params
print(f'Best Param (numTrees): {rfw_model.stages[-1].getNumTrees}')
print(f'Best Param (maxDepth): {rfw_model.stages[-1].getMaxDepth()}')
print(f'Best Param (featureSubsetStrategy): {rfw_model.stages[-1].getFeatureSubsetStrategy()}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evalute results in the training set

# COMMAND ----------

# evaluate results in the training set
predictions = rfw_model.transform(trainDF.filter(trainDF.DEP_DEL15.isNotNull()))
eval_ = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='DEP_DEL15')

# store performance metrics in a dictionary
metrics = ['accuracy', 'weightedPrecision', 'weightedRecall', 'weightedFMeasure', 'precisionByLabel', 'recallByLabel', 'fMeasureByLabel']
results = {}
for m in metrics:
  if m in ['precisionByLabel', 'recallByLabel', 'fMeasureByLabel']:
    results[m] = [eval_.evaluate(predictions, {eval_.metricName: m, eval_.metricLabel:0.0}), 
                  eval_.evaluate(predictions, {eval_.metricName: m, eval_.metricLabel:1.0})]
  else:
    results[m] = eval_.evaluate(predictions, {eval_.metricName: m})

# print results
print('Performance metrics - TRAINING SET')
print('------------------------------------------------------------------------------------------------')
for x in results:
  print(f'{x}: {results[x]}')
  
# save results
rfw_train = results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate results in the test set

# COMMAND ----------

# evaluate results in the test set
predictions = rfw_model.transform(testDF)
eval_ = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='DEP_DEL15')

# store performance metrics in a dictionary
metrics = ['accuracy', 'weightedPrecision', 'weightedRecall', 'weightedFMeasure', 'precisionByLabel', 'recallByLabel', 'fMeasureByLabel']
results = {}
for m in metrics:
  if m in ['precisionByLabel', 'recallByLabel', 'fMeasureByLabel']:
    results[m] = [eval_.evaluate(predictions, {eval_.metricName: m, eval_.metricLabel:0.0}), 
                  eval_.evaluate(predictions, {eval_.metricName: m, eval_.metricLabel:1.0})]
  else:
    results[m] = eval_.evaluate(predictions, {eval_.metricName: m})

# print results
print('Performance metrics - TEST SET')
print('------------------------------------------------------------------------------------------------')
for x in results:
  print(f'{x}: {results[x]}')
  
# save results
rfw_test = results

# COMMAND ----------

# MAGIC %md
# MAGIC # Linear SVM

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set the pipeline

# COMMAND ----------

# define categorical and continuous variables
categoricals = ['QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'DEP_HOUR', 'OP_CARRIER', 'ORIGIN', 'DEST', 'Bad_Weather_Prediction']
numerics = ['rolling_average', 'delay_2hrs_originport', 'delay_4hrs_originport', 'delay_8hrs_originport', 'delay_12hrs_originport', 'delay_2hrs_destport', 'delay_4hrs_destport', 'delay_8hrs_destport', 'delay_12hrs_destport', 'delay_2hrs_orgairline', 'delay_4hrs_orgairline', 'delay_8hrs_orgairline', 'delay_12hrs_orgairline', 'arrdelay_2hrs_originport', 'arrdelay_4hrs_originport', 'arrdelay_8hrs_originport', 'arrdelay_12hrs_originport', 'delay_2hrs_originhub', 'delay_4hrs_originhub', 'delay_8hrs_originhub', 'delay_12hrs_originhub', 'WND_direction_angle', 'WND_speed', 'CIG_ceiling_height_dimension', 'VIS_distance_dimension', 'SLP_sea_level_pressure', 'TMP_air_temperature', 'DEW_dew_point_temperature']

# define feature transformations
indexer = map(lambda c: StringIndexer(inputCol=c, outputCol=c+'_idx', handleInvalid = 'keep'), categoricals)
ohes = map(lambda c: OneHotEncoder(inputCol=c+'_idx', outputCol=c+'_class'), categoricals)
imputer = Imputer(strategy='median', inputCols = numerics, outputCols = numerics)
feature_cols = list(map(lambda c: c+'_idx', categoricals)) + numerics
vassembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures',
                        withStd=True, withMean=True)

# create a map of {idx: count of distinct values} for the categorial features
cat_features_map = {idx: trainDF.select(categoricals[idx]).distinct().count() for idx in range(len(categoricals))}
max_dist_values = sorted(cat_features_map.values())[-1]+1

# set-up the algorithm
svm = LinearSVC(featuresCol='scaledFeatures', labelCol='DEP_DEL15', 
                maxIter=100, regParam=0.1, standardization=False)

# assemble the pipeline
svm_transf_stages = list(indexer) + list(ohes) + [imputer] + [vassembler] + [scaler] + [svm]
svm_pipeline = Pipeline(stages=svm_transf_stages)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select best model using TimeSeriesCrossValidator

# COMMAND ----------

# build the parameter grid for model tuning
svm_paramGrid = ParamGridBuilder() \
               .addGrid(svm.maxIter, [50, 100]) \
               .addGrid(svm.regParam, [0.01, 0.1, 1.0]) \
               .build()

# execute TimeSeriesCrossValidator for model tuning
svm_crossval = TimeSeriesCrossValidator(estimator=svm_pipeline,
                          estimatorParamMaps=svm_paramGrid,
                          evaluator=BinaryClassificationEvaluator(labelCol='DEP_DEL15', 
                                                                  metricName='areaUnderROC'),
                          parallelism=3,
                          foldCol='foldCol',
                          numFolds=nFolds)

# train the tuned model and establish our best model
start = time.time()
svm_cvModel = svm_crossval.fit(foldedTrainDF)
svm_model = svm_cvModel.bestModel
print(f'CV-training time: {time.time() - start} seconds')
print('')

# print best model params
print(f'Best Param (maxIter): {svm_model.stages[-1].getMaxIter()}')
print(f'Best Param (regParam): {svm_model.stages[-1].getRegParam()}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evalute results in the training set

# COMMAND ----------

# evaluate results in the training set
predictions = svm_model.transform(trainDF.filter(trainDF.DEP_DEL15.isNotNull()))
eval_ = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='DEP_DEL15')

# store performance metrics in a dictionary
metrics = ['accuracy', 'weightedPrecision', 'weightedRecall', 'weightedFMeasure', 'precisionByLabel', 'recallByLabel', 'fMeasureByLabel']
results = {}
for m in metrics:
  if m in ['precisionByLabel', 'recallByLabel', 'fMeasureByLabel']:
    results[m] = [eval_.evaluate(predictions, {eval_.metricName: m, eval_.metricLabel:0.0}), 
                  eval_.evaluate(predictions, {eval_.metricName: m, eval_.metricLabel:1.0})]
  else:
    results[m] = eval_.evaluate(predictions, {eval_.metricName: m})

# print results
print('Performance metrics - TRAINING SET')
print('------------------------------------------------------------------------------------------------')
for x in results:
  print(f'{x}: {results[x]}')

# save results
svm_train = results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate results in the test set

# COMMAND ----------

# evaluate results in the test set
predictions = svm_model.transform(testDF)
eval_ = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='DEP_DEL15')

# store performance metrics in a dictionary
metrics = ['accuracy', 'weightedPrecision', 'weightedRecall', 'weightedFMeasure', 'precisionByLabel', 'recallByLabel', 'fMeasureByLabel']
results = {}
for m in metrics:
  if m in ['precisionByLabel', 'recallByLabel', 'fMeasureByLabel']:
    results[m] = [eval_.evaluate(predictions, {eval_.metricName: m, eval_.metricLabel:0.0}), 
                  eval_.evaluate(predictions, {eval_.metricName: m, eval_.metricLabel:1.0})]
  else:
    results[m] = eval_.evaluate(predictions, {eval_.metricName: m})

# print results
print('Performance metrics - TEST SET')
print('------------------------------------------------------------------------------------------------')
for x in results:
  print(f'{x}: {results[x]}')
  
# save results
svm_test = results

# COMMAND ----------

# MAGIC %md
# MAGIC # Gradient Boosting Trees

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set the pipeline

# COMMAND ----------

# define categorical and continuous variables
categoricals = ['QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'DEP_HOUR', 'OP_CARRIER', 'ORIGIN', 'DEST', 'Bad_Weather_Prediction']
numerics = ['rolling_average', 'delay_2hrs_originport', 'delay_4hrs_originport', 'delay_8hrs_originport', 'delay_12hrs_originport', 'delay_2hrs_destport', 'delay_4hrs_destport', 'delay_8hrs_destport', 'delay_12hrs_destport', 'delay_2hrs_orgairline', 'delay_4hrs_orgairline', 'delay_8hrs_orgairline', 'delay_12hrs_orgairline', 'arrdelay_2hrs_originport', 'arrdelay_4hrs_originport', 'arrdelay_8hrs_originport', 'arrdelay_12hrs_originport', 'delay_2hrs_originhub', 'delay_4hrs_originhub', 'delay_8hrs_originhub', 'delay_12hrs_originhub', 'WND_direction_angle', 'WND_speed', 'CIG_ceiling_height_dimension', 'VIS_distance_dimension', 'SLP_sea_level_pressure', 'TMP_air_temperature', 'DEW_dew_point_temperature']

# define feature transformations
indexer = map(lambda c: StringIndexer(inputCol=c, outputCol=c+'_idx', handleInvalid = 'keep'), categoricals)
ohes = map(lambda c: OneHotEncoder(inputCol=c+'_idx', outputCol=c+'_class'), categoricals)
imputer = Imputer(strategy='median', inputCols = numerics, outputCols = numerics)
feature_cols = list(map(lambda c: c+'_idx', categoricals)) + numerics
vassembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures',
                        withStd=True, withMean=True)

# create a map of {idx: count of distinct values} for the categorial features
cat_features_map = {idx: trainDF.select(categoricals[idx]).distinct().count() for idx in range(len(categoricals))}
max_dist_values = sorted(cat_features_map.values())[-1]+1

# set-up the algorithm
gbt = GBTClassifier(featuresCol='features', labelCol='DEP_DEL15', maxIter=50, stepSize=0.1, maxDepth=1, 
                    maxBins=max_dist_values, featureSubsetStrategy='sqrt', seed=123)

# assemble the pipeline
gbt_transf_stages = list(indexer) + [imputer] + [vassembler] + [gbt]
gbt_pipeline = Pipeline(stages=gbt_transf_stages)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select best model using TimeSeriesCrossValidator

# COMMAND ----------

# build the parameter grid for model tuning
gbt_paramGrid = ParamGridBuilder() \
               .addGrid(gbt.stepSize, [0.1, 1.0]) \
               .addGrid(gbt.maxDepth, [2, 4]) \
               .build()

# execute TimeSeriesCrossValidator for model tuning
gbt_crossval = TimeSeriesCrossValidator(estimator=gbt_pipeline,
                          estimatorParamMaps=gbt_paramGrid,
                          evaluator=BinaryClassificationEvaluator(labelCol='DEP_DEL15', 
                                                                  metricName='areaUnderROC'),
                          parallelism=3,
                          foldCol='foldCol',
                          numFolds=nFolds)

# train the tuned model and establish our best model
start = time.time()
gbt_cvModel = gbt_crossval.fit(foldedTrainDF)
gbt_model = gbt_cvModel.bestModel
print(f'CV-training time: {time.time() - start} seconds')
print('')

# print best model params
print(f'Best Param (maxIter): {gbt_model.stages[-1].getMaxIter()}')
print(f'Best Param (stepSize): {gbt_model.stages[-1].getStepSize()}')
print(f'Best Param (maxDepth): {gbt_model.stages[-1].getMaxDepth()}')
print(f'Best Param (featureSubsetStrategy): {gbt_model.stages[-1].getFeatureSubsetStrategy()}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evalute results in the training set

# COMMAND ----------

# evaluate results in the training set
predictions = gbt_model.transform(trainDF.filter(trainDF.DEP_DEL15.isNotNull()))
eval_ = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='DEP_DEL15')

# store performance metrics in a dictionary
metrics = ['accuracy', 'weightedPrecision', 'weightedRecall', 'weightedFMeasure', 'precisionByLabel', 'recallByLabel', 'fMeasureByLabel']
results = {}
for m in metrics:
  if m in ['precisionByLabel', 'recallByLabel', 'fMeasureByLabel']:
    results[m] = [eval_.evaluate(predictions, {eval_.metricName: m, eval_.metricLabel:0.0}), 
                  eval_.evaluate(predictions, {eval_.metricName: m, eval_.metricLabel:1.0})]
  else:
    results[m] = eval_.evaluate(predictions, {eval_.metricName: m})

# print results
print('Performance metrics - TRAINING SET')
print('------------------------------------------------------------------------------------------------')
for x in results:
  print(f'{x}: {results[x]}')
  
# save results
gbt_train = results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate results in the test set

# COMMAND ----------

# evaluate results in the test set
predictions = gbt_model.transform(testDF)
eval_ = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='DEP_DEL15')

# store performance metrics in a dictionary
metrics = ['accuracy', 'weightedPrecision', 'weightedRecall', 'weightedFMeasure', 'precisionByLabel', 'recallByLabel', 'fMeasureByLabel']
results = {}
for m in metrics:
  if m in ['precisionByLabel', 'recallByLabel', 'fMeasureByLabel']:
    results[m] = [eval_.evaluate(predictions, {eval_.metricName: m, eval_.metricLabel:0.0}), 
                  eval_.evaluate(predictions, {eval_.metricName: m, eval_.metricLabel:1.0})]
  else:
    results[m] = eval_.evaluate(predictions, {eval_.metricName: m})

# print results
print('Performance metrics - TEST SET')
print('------------------------------------------------------------------------------------------------')
for x in results:
  print(f'{x}: {results[x]}')

# save results
gbt_test = results

# COMMAND ----------

# MAGIC %md
# MAGIC # Performance comparison

# COMMAND ----------

# MAGIC %md
# MAGIC Here we compare the algorithms performance after hyperparameter tuning through TimeSeriesCV.

# COMMAND ----------

# plot best model scores
fig = plt.figure(figsize=(12, 4))
classifiers = ['LR', 'RF', 'SVM', 'GBT']
metrics = ['recallByLabel', 'fMeasureByLabel', 'precisionByLabel']
titles = ['Recall\n(label 1 = delay)', 'fMeasure\n(label 1 = delay)', 'Precision\n(label = delay)']
nplots = len(metrics)
results = [
  [lr_test[metrics[0]][1],rfw_test[metrics[0]][1],svm_test[metrics[0]][1],gbt_test[metrics[0]][1]],
  [lr_test[metrics[1]][1],rfw_test[metrics[1]][1],svm_test[metrics[1]][1],gbt_test[metrics[1]][1]],
  [lr_test[metrics[2]][1],rfw_test[metrics[2]][1],svm_test[metrics[2]][1],gbt_test[metrics[2]][1]],
]
for idx in range(nplots):
  ax = plt.subplot(1, nplots, idx+1)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.bar(classifiers, results[idx], alpha=0.5)
  ax.set_ylim(0.0, 1.0)
  ax.set_title(titles[idx], pad=20)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusions

# COMMAND ----------

# MAGIC %md
# MAGIC On our main metric of interest (Recall), the best-performing algorithms were Random Forest and GBT, with Recall slightly above 80%. In terms of fMeasure (out tie-break metric), Gradient Boosting Tree had a slightly better performance, due to its higher Precision compared to Random Forest. So the winner is GBT. GBT is expensive to train, but since we don't antecipate the need for constant retraining this should not be a concern. Prediction time on the other hand takes slighly longer on ensemble models like GBT or RF than on not-emsemble models such as SVM or Logistic Regression. If performance at prediction time is important, this consideration could make us opt for SVM instead, which is as strong as in terms of Recall but slightly less performant in terms of fMeasure and Precision.
