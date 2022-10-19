# Databricks notebook source
# MAGIC %md
# MAGIC # Algorithm Implementation

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook explains the math behind Regularized Logistic Regression and implements it from scratch in a distributed way using Spark RDD.
# MAGIC 
# MAGIC We apply the algorithm to a small sample of the data (3 months) and then compare the results with the Logistic Regression method available at the pyspark.ml library.
# MAGIC 
# MAGIC <a href='$./fp_main_notebook_final'>To return to main notebook click here</a>.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Logistic function

# COMMAND ----------

# MAGIC %md
# MAGIC In binary logistic regression, our response variable \\(Y\\) is categorical and can assume only two values: 1 or 0 (e.g. a flight delays more than 15 minutes or not).
# MAGIC 
# MAGIC Our predictions \\(\hat{y}\\) are a function of \\(p(\mathbf{x})\\), the conditional probability that a flight delays given \\(X=\mathbf{x}\\):
# MAGIC 
# MAGIC $$p(\mathbf{x})=Pr(Y=1|X=\mathbf{x})$$
# MAGIC 
# MAGIC \\(p(\mathbf{x})\\) is a probability, and so can assume any real value between [0,1].
# MAGIC 
# MAGIC To complete the classification task, we need to assume a threshold (e.g. 0.5) to map the predicted probability \\(p(\mathbf{x})\\) to one of the actual labels \\(\hat{y}\\) can assume (1 or 0):
# MAGIC 
# MAGIC $$
# MAGIC \begin{dcases} 
# MAGIC   \hat{y}=1 &\text{if } p(\mathbf{x})\ge0.5 \\\
# MAGIC   \hat{y}=0 &\text{if } p(\mathbf{x})<0.5 
# MAGIC \end{dcases}
# MAGIC $$
# MAGIC 
# MAGIC To ensure \\(p(\mathbf{x})\\) is always bounded between [0,1], we model it using the sigmoid function:
# MAGIC 
# MAGIC $$p(\mathbf{x}) = \frac{1}{1+e^{-(w_0x_0+w_1x_1+...+w_px_p)}}$$ 
# MAGIC 
# MAGIC where \\(\mathbf{w} = w_0, w_1, ..., w_p\\) is the weight vector of our model, and \\(\mathbf{x} = x_0, x_1, ..., x_p \\) is our predictor vector. Note that we are using an 'augmented' notation for our weight and predictor vectors, where \\(w_0\\) refers to the bias term and \\(x_0\\) is always set equal to 1.
# MAGIC 
# MAGIC After a bit of algebraic manipulation and after taking the logarithm of both sides we arrive at:
# MAGIC 
# MAGIC $$log\biggl(\frac{p(\mathbf{x})}{1-p(\mathbf{x})}\biggl)=w_0x_0+w_1x_1+...+w_px_p$$
# MAGIC 
# MAGIC This equation shows that the log odds of \\(p(\mathbf{x})\\) is linear in \\(\mathbf{x}\\).

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Log Loss function regularized using Elastic Net

# COMMAND ----------

# MAGIC %md
# MAGIC For every parametric machine learning algorithm, we need to define a loss function which we want to minimize in order to determine the optimal parameters \\(\mathbf\{w}\\) of our model.
# MAGIC 
# MAGIC In the case of logistic regression, we use the Log Loss function. We added to it a regularization term, having opted to use Elastic net, a hybrid of L1 and L2 regularizations, which is the default implementation in the pyparsk MLLib and gives us flexibility to experiment with different regularizations as a function of hyperparameter \\(\alpha\\).
# MAGIC 
# MAGIC Mathematically, we have:
# MAGIC 
# MAGIC $$l(\mathbf{w})=-\frac{1}{n}\sum_{i=1}^n\biggl[y^{(i)}\log(p(\mathbf{x})^{(i)})+(1-y^{(i)})\log(1-p(\mathbf{x})^{(i)})\biggl]+\alpha\biggl(\frac{\lambda}{n}||\mathbf{w}||_1\biggl)+(1-\alpha)\biggl(\frac{\lambda}{2n}||\mathbf{w}||_2^2\biggl), \alpha\in[0,1], \lambda\ge0$$

# COMMAND ----------

# MAGIC %md
# MAGIC ## The gradients of regularized Log Loss function

# COMMAND ----------

# MAGIC %md
# MAGIC To apply the Gradient Descent algorithm, we need to compute the gradient of our loss function.
# MAGIC 
# MAGIC The gradient for our weight vector in all positions except the bias term (\\(w_0\\)) is defined as:
# MAGIC 
# MAGIC $$\nabla_{\mathbf{w_k}}=\frac{1}{n}\biggl(p(\mathbf{x}^{(i)})-y^{(i)}\biggl)*\mathbf{x}+\frac{\alpha\lambda}{n} sign(\mathbf{w})+\frac{(1-\alpha)\lambda}{n}\mathbf{w}, k \in [1,2,...p]$$
# MAGIC 
# MAGIC For the bias term we have:
# MAGIC 
# MAGIC $$\nabla_{w_0}=\frac{1}{n}\biggl(p(\mathbf{x}^{(i)})-y^{(i)}\biggl)$$

# COMMAND ----------

# MAGIC %md
# MAGIC # Algorithm implementation

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

# Read clean dataset
rawDataDF = spark.read.parquet(f'{blob_url}/airlines_data_latest_weather_3m_trimmed_v4')

# Print out number of rows
print(str(rawDataDF.count()) + ' rows in the data.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filter columns

# COMMAND ----------

# MAGIC %md
# MAGIC Filter only columns of interest.

# COMMAND ----------

# label and features of interest
cols = ['DEP_DEL15', 'YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'OP_CARRIER', 'ORIGIN', 'DEST', 'rolling_average', 'delay_2hrs_originport', 'delay_4hrs_originport', 'delay_8hrs_originport', 'delay_12hrs_originport', 'delay_2hrs_destport', 'delay_4hrs_destport', 'delay_8hrs_destport', 'delay_12hrs_destport', 'delay_2hrs_orgairline', 'delay_4hrs_orgairline', 'delay_8hrs_orgairline', 'delay_12hrs_orgairline', 'arrdelay_2hrs_originport', 'arrdelay_4hrs_originport', 'arrdelay_8hrs_originport', 'arrdelay_12hrs_originport', 'DEP_HOUR', 'Part_of_Day', 'WND_direction_angle', 'WND_speed', 'CIG_ceiling_height_dimension', 'VIS_distance_dimension', 'TMP_air_temperature', 'DEW_dew_point_temperature', 'SLP_sea_level_pressure']

# filter cols of interest
filteredDataDF = rawDataDF.select(cols)

# print out number of features
# minus two to account for the response and the year variables (which will only be used for splitting)
print(str(len(filteredDataDF.columns)-2) + ' features in the data.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split train and test data

# COMMAND ----------

# set 2019 as the hold-out test set
hold_out_variable = 'MONTH'
hold_out_threshold = '3'

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
augmentedTrainDF = major_df.unionAll(oversample_df)
augmentedTrainDF.groupBy('DEP_DEL15').count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set the pipeline

# COMMAND ----------

# define categorical and continuous variables
categoricals = ['DAY_OF_MONTH', 'DAY_OF_WEEK', 'DEP_HOUR', 'OP_CARRIER', 'ORIGIN', 'DEST', 'TMP_air_temperature', 'DEW_dew_point_temperature']
numerics = ['delay_2hrs_originport', 'delay_4hrs_originport', 'delay_8hrs_originport', 'delay_12hrs_originport', 'delay_2hrs_destport', 'delay_4hrs_destport', 'delay_8hrs_destport', 'delay_12hrs_destport', 'delay_2hrs_orgairline', 'delay_4hrs_orgairline', 'delay_8hrs_orgairline', 'delay_12hrs_orgairline', 'arrdelay_2hrs_originport', 'arrdelay_4hrs_originport', 'arrdelay_8hrs_originport', 'arrdelay_12hrs_originport', 'WND_direction_angle', 'WND_speed', 'CIG_ceiling_height_dimension', 'VIS_distance_dimension', 'SLP_sea_level_pressure']

# define feature transformations
indexer = map(lambda c: StringIndexer(inputCol=c, outputCol=c+'_idx', handleInvalid = 'keep'), categoricals)
ohes = map(lambda c: OneHotEncoder(inputCol=c+'_idx', outputCol=c+'_class'), categoricals)
imputer = Imputer(strategy='median', inputCols = numerics, outputCols = numerics)
feature_cols = list(map(lambda c: c+'_idx', categoricals)) + numerics
vassembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures',
                        withStd=True, withMean=True)

# assemble the pipeline
lr_transf_stages = list(indexer) + list(ohes) + [imputer] + [vassembler] + [scaler]
lr_pipeline = Pipeline(stages=lr_transf_stages)

# transform the data
transfTrainDF = lr_pipeline.fit(augmentedTrainDF).transform(augmentedTrainDF).select(['scaledfeatures', 'DEP_DEL15'])
transfTestDF = lr_pipeline.fit(augmentedTrainDF).transform(testDF).select(['scaledfeatures', 'DEP_DEL15'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cast the data as RDD

# COMMAND ----------

# Cast as RDD where records are tuples of (features_array, y)
trainRDD = transfTrainDF.rdd.map(lambda x: (np.array(x[:-1]), x[-1])).cache()
testRDD = transfTestDF.rdd.map(lambda x: (np.array(x[:-1]), x[-1])).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize baseline model

# COMMAND ----------

num_features = trainRDD.map(lambda x: x[0].shape[1]).take(1)
mean_prob_delay = trainRDD.map(lambda x: x[1]).mean()
log_odds = np.log(mean_prob_delay/(1-mean_prob_delay))
wInit = np.concatenate((np.array([log_odds]), np.zeros(num_features)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Sigmoid function

# COMMAND ----------

# helper function to compute the probability of positive class
def Sigmoid(dataRDD, W):
    """
    Compute the conditional probability P(y = 1 | X; W).
    Args:
        dataRDD - each record is a tuple of (features_array, y)
        W       - (array) model coefficients with bias at index 0
    Returns:
        predRDD - records are tuples of (features_array (true_y, prob_y))
    """
    predRDD = dataRDD.map(lambda x: (np.append([1.0], x[0]), x[1])) \
                     .map(lambda x: (x[0], (x[1], np.power(1.0 + np.exp(-W.dot(x[0])), -1))))
    return predRDD

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Predict function

# COMMAND ----------

# helper function to predict class
def Predict(dataRDD, W, threshold=0.5):
    """
    Helper function for predicting class.
    Args:
        dataRDD - each record is a tuple of (features_array, y)
        W       - (array) model coefficients with bias at index 0
        threshold - (float) minimum probability to assign class 1; defaults to 0.5
    Returns:
        predRDD - records are tuples of (features_array (true_y, pred_y))
    """
    predRDD = Sigmoid(dataRDD, W).mapValues(lambda x: (x[0], 1 if x[1]>=threshold else 0))
    return predRDD

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Regularized Log Loss function

# COMMAND ----------

# helper function to compute regularized log loss
def LogLoss_wReg(dataRDD, W, elasticNetParam = 0.5, regParam = 0.1):
    """
    Compute regularized log loss error.
    Args:
        dataRDD - each record is a tuple of (features_array, y)
        W       - (array) model coefficients with bias at index 0
    """
    logloss, n = Sigmoid(dataRDD, W).map(lambda x: (x[1][0]*np.log(x[1][1])+(1-x[1][0])*np.log(1-x[1][1]), 1)) \
                                    .reduce(lambda x,y: (x[0]+y[0], x[1]+y[1]))
    lossterm = -logloss/n  
    np.append([0.0], W[1:]) # the bias is not included in the regularization term
    regterm = elasticNetParam*regParam/n*np.absolute(W).sum() + (1-elasticNetParam)*regParam/(2*n)*W.dot(W)
    logloss_wReg = lossterm + regterm
    return logloss_wReg

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Gradient Descent with Elastic Net Regularization

# COMMAND ----------

# helper function to perform one gradient descent step with regularization
def GDUpdate_wReg(dataRDD, W, learningRate = 0.1, elasticNetParam = 0.5, regParam = 0.1):
    """
    Perform one gradient descent step/update with elastic net regularization.
    Args:
        dataRDD - tuple of (features_array, y)
        W       - (array) model coefficients with bias at index 0
        learningRate - (float) defaults to 0.1
        elasticNetParam - (float) defaults to 0.5
        regParam - (float) regularization term coefficient
    Returns:
        model   - (array) updated coefficients, bias still at index 0
    """
    # unregularized gradient
    unreg_grad, n = Sigmoid(dataRDD, W).map(lambda x: (x[0]*(x[1][1]-x[1][0]), 1)) \
                                       .reduce(lambda x,y: (x[0]+y[0], x[1]+y[1]))
    # regularized gredient
    W_ = np.append([0.0], W[1:]) # gradient of regularization term doesn't apply to bias
    reg_grad = unreg_grad/n + elasticNetParam*regParam*np.sign(W_)/n + (1-elasticNetParam)*regParam*W_/n
    
    # update model
    new_model = W - learningRate*reg_grad
    
    return new_model

# COMMAND ----------

# helper function to compute gradient descent function
def GradientDescent_wReg(trainRDD, testRDD, wInit, nSteps = 30, learningRate = 0.1,
                         elasticNetParam = 0.5, regParam = 0.1, verbose = False):
    """
    Perform nSteps iterations of regularized gradient descent and 
    track loss on a test and train set. Return lists of
    test/train loss and the models themselves.
    """
    # initialize lists to track model performance
    train_history, test_history, model_history = [], [], []
    
    # perform n updates & compute test and train loss after each epoch
    model = wInit
    for idx in range(nSteps):  
        # update the model
        model = GDUpdate_wReg(trainRDD, model, learningRate, elasticNetParam, regParam)
        
        # keep track of test/train loss for plotting
        training_loss = LogLoss_wReg(trainRDD, model, elasticNetParam, regParam)
        test_loss = LogLoss_wReg(testRDD, model, elasticNetParam, regParam)
        train_history.append(training_loss)
        test_history.append(test_loss)
        model_history.append(model)
        
        # console output if desired
        if verbose:
            print('-'*50)
            print(f'STEP: {idx+1}')
            print(f'training loss: {training_loss}')
            print(f'test loss: {test_loss}')
            print(f'Model: {[w for w in model]}')
            
    return train_history, test_history, model_history

# COMMAND ----------

# helper function to plot error curves
def plotErrorCurves(trainLoss, testLoss, title = None):
    """
    Helper function for plotting.
    Args: trainLoss (list of losses) , testLoss (list of losses)
    """
    fig, ax = plt.subplots(1,1,figsize=(16,8))
    x = list(range(len(trainLoss)+1))[1:]
    ax.plot(x, trainLoss, 'k--', label='Training Loss')
    ax.plot(x, testLoss, 'r--', label='Test Loss')
    ax.legend(loc='upper right', fontsize='x-large')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Log Loss')
    if title:
        plt.title(title)
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Confusion Matrix

# COMMAND ----------

# helper function to compute the confusion matrix
def ConfusionMatrix(dataRDD, W, threshold=0.5):
  """
  Helper function to compute the confusion matrix.
  Args:
        dataRDD - each record is a tuple of (features_array, y)
        W       - (array) model coefficients with bias at index 0
        threshold - (float) minimum probability to assign class 1; defaults to 0.5
    Returns:
        ConfMatrix - (list) of [TP,FP,FN,TN]
  """
  
  def classify(row):
    """Helper function to perform classification row by row"""
    if row[1]==1: #predicted class is 1
      if row[0]==1: #actual class is 1
        return 'TP'
      if row[0]==0: #actual class is 0
        return 'FP'
    if row[1]==0: #predicted class is 0
      if row[0]==1: #actual class is 1
        return 'FN'
      if row[0]==0: #actual class is 0
        return 'TN'
  
  ConfMatrix = Predict(dataRDD, W, threshold).map(lambda x: (classify(x[1]),1)) \
                                             .reduceByKey(lambda x,y: x+y) \
                                             .collect()
  return ConfMatrix

# COMMAND ----------

# helper function to compute performance metrics
def metrics(dataRDD, W, threshold=0.5):
  """
  Helper function to compute the confusion matrix.
  Args:
        dataRDD - each record is a tuple of (features_array, y)
        W       - (array) model coefficients with bias at index 0
        threshold - (float) minimum probability to assign class 1; defaults to 0.5
    Returns:
        perf_metrics - (list) of Accuracy, Weighted Precision, Weighted Recall,
                       Weighted F-Score, Precision By Label, Recall By Label,
                       F-Score By Label.
  """ 
  cm = dict(ConfusionMatrix(dataRDD, W, threshold))
  n = cm.get('TP',0)+cm.get('FP',0)+cm.get('FN',0)+cm.get('TN',0)
  Accuracy = (cm.get('TP',0)+cm.get('TN',0))/n
  PrecisionByLabel = [cm.get('TN',0)/(cm.get('FN',0)+cm.get('TN',0)), cm.get('TP',0)/(cm.get('TP',0)+cm.get('FP',0))]
  RecallByLabel = [cm.get('TN',0)/(cm.get('FP',0)+cm.get('TN',0)), cm.get('TP',0)/(cm.get('TP',0)+cm.get('FN',0))]
  WeightedPrecision = PrecisionByLabel[0]*(cm.get('FN',0)+cm.get('TN',0))/n + PrecisionByLabel[1]*(cm.get('TP',0)+cm.get('FP',0))/n
  WeightedRecall = RecallByLabel[0]*(cm.get('FP',0)+cm.get('TN',0))/n + RecallByLabel[1]*(cm.get('TP',0)+cm.get('FN',0))/n
  FScoreByLabel = list(2*np.array(PrecisionByLabel)*np.array(RecallByLabel)/(np.array(PrecisionByLabel)+np.array(RecallByLabel)))
  WeightedFScore = 2*WeightedPrecision*WeightedRecall/(WeightedPrecision+WeightedRecall)
  perf_metrics = [Accuracy, WeightedPrecision, WeightedRecall, WeightedFScore,
                 PrecisionByLabel, RecallByLabel, FScoreByLabel]
  
  return perf_metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results (from scratch implementation)

# COMMAND ----------

start = time.time()
train_history, test_history, model_history = GradientDescent_wReg(trainRDD, testRDD, wInit, nSteps = 50, learningRate = 0.1, 
                                                                  elasticNetParam = 0.5, regParam = 0.1, verbose = False)
print(f'Training time: {time.time() - start} seconds')
results = metrics(testRDD, W, threshold=0.5)

# COMMAND ----------

print('Performance metrics')
print('------------------------------------------------------------------------------------------------')
print(f'Accuracy: {results[0]}')
print(f'Weighted Precision: {results[1]}')
print(f'Weighted Recall: {results[2]}')
print(f'F-Score: {results[3]}')
print(f'Precision By Label: {results[4]}')
print(f'Recall By Label: {results[5]}')
print(f'F-Score by Label: {results[6]}')
plotErrorCurves(train_history, test_history)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare results with pyspark.ml.classification.LogisticRegression implementation

# COMMAND ----------

# initialize Logistic Regression
lr = LogisticRegression(featuresCol='scaledfeatures', labelCol='DEP_DEL15', maxIter=50, regParam=0.1, elasticNetParam=0.5, 
                        standardization=False, family='binomial')

# fit the model
lr_model = lr.fit(transfTrainDF)

# assess performance metrics
lr_summary = lr_model.evaluate(transfTestDF)

# COMMAND ----------

print('Performance metrics')
print('------------------------------------------------------------------------------------------------')
print(f'Accuracy: {lr_summary.accuracy}')
print(f'Weighted Precision: {lr_summary.weightedPrecision}')
print(f'Weighted Recall: {lr_summary.weightedRecall}')
print(f'F-Score: {lr_summary.weightedFMeasure()}')
print(f'Precision By Label: {lr_summary.precisionByLabel}')
print(f'Recall By Label: {lr_summary.recallByLabel}')
print(f'F-Score by Label: {lr_summary.fMeasureByLabel()}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusions

# COMMAND ----------

# MAGIC %md
# MAGIC Our own implementation had a worst performance when compared to the pyspark.ml.classification.LogisticRegression implementation. We got a delay recall of 0.43 (vs 0.65 of pyspark implementation) and a delay precision of 0.29 (vs. 0.57 of pyspark implementation). We tried several adjustments to match the results but the differences continued to be large. We tried to keep the parameter sets the most comparable we could but there are many more parameters in the LogisticRegression class than in our own implementation. We are not sure as well that the pyspark implementation do use Batch Gradient Descent as the optimization algorithm under the hood. If it uses other algorithms (e.g. SGD or LBFGS) than it is understandable that results will be less comparable. The learning rate parameter is also a parameter that is not transparent in the pyspark implementation.
