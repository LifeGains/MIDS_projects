# Databricks notebook source
# MAGIC %md
# MAGIC # Flight's On-Time Performance Data
# MAGIC 
# MAGIC This notebook contains On-time performance of different airlines from 2015 - 2019. In this notebook we present Exploratory Data Analysis (EDA), feature engineering and creation on this dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Storage and Import

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set Up The Blob 

# COMMAND ----------

# Init script to create the blob URL
# Put this at the top of every notebook
from pyspark.sql.functions import col, max

blob_container = "team07" # The name of your container created in https://portal.azure.com
storage_account = "team07" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team07" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team07" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

# Generates the SAS token
# Put this at the top of every notebook
spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

# Source: https://ucbischool.slack.com/archives/C02C3SFLC11/p1635569501096100?thread_ts=1635526204.076500&cid=C02C3SFLC11 
# Displays what is currently in the blob
# Put this at the top of every notebook
display(dbutils.fs.ls(blob_url))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run Imports

# COMMAND ----------

pip install timezonefinder

# COMMAND ----------

pip install geopy

# COMMAND ----------

#ALL IMPORTS
import re
import pandas as pd
import numpy as np
import seaborn as sns
import pyspark
import matplotlib.pyplot as plt
from pyspark.sql.functions import col
from pyspark.sql import functions as sf
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from datetime import datetime 
from pyspark.sql.functions import split
import pytz
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat,col
from pyspark.sql.functions import col
from pyspark.sql.types import TimestampType
from pyspark.sql.types import StringType
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col,sum
from pyspark.sql.functions import desc
from pyspark.sql.functions import rank
from pyspark.sql.window import Window
from pyspark.sql.functions import col,sum

# COMMAND ----------

# Inspect the Mount's Final Project folder 
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Full Airlines Dataset

# COMMAND ----------

#Load all Flights Data
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*")
#display(df_airlines)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## EDA AIRLINES DATA

# COMMAND ----------

# MAGIC %md
# MAGIC #### Identify and De-duplicate data
# MAGIC On our initial exploration we found that the airlines dataset had over 63 million rows but were duplicated. We dropped these duplicates and the final de-duplicated count for airlines dataset was ~ 31 million. We will be working on this new de-duplicated dataset from this point.

# COMMAND ----------

#Original count
row_org = df_airlines.count()
print("Original row count", row_org)

#Identify duplicates in airlines dataset
distinctDF = df_airlines.distinct()
print("Distinct count: "+str(distinctDF.count()))

#Drop duplicate rows identified
df_airlines = df_airlines.dropDuplicates()
row = df_airlines.count()
print("De-duplicated row count", row)

# COMMAND ----------

# extracting number of columns from the Dataframe
column = len(df_airlines.columns)
column

# COMMAND ----------

# MAGIC %md
# MAGIC #### General Descriptives and Frequencies for columns of interest

# COMMAND ----------

# MAGIC %md
# MAGIC We ran descriptive statistics on some columns of interest to understand flight characteristics, delayed times and reasons for delay. On running these some highlight results were that Late aircraft and Carrier delay reasons have highest average delay in minutes compared across other reasons. Most flights were short distance (~800 miles) and roughly little over 2 hours, average delay in departure and arrival was roughly 12 minutes. This is in line with our definition of delayed flights for this project where any flight with delay > 15 minutes is regarded as delayed.

# COMMAND ----------

# Descriptive statistics on some columns of interest to understand flight characteristics, delayed times and reasons for delay
df_airlines.select(['DEP_DELAY_NEW', 'ARR_DELAY_NEW','TAXI_OUT', 'AIR_TIME', 'ACTUAL_ELAPSED_TIME',
           'DISTANCE','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']).describe().toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC - On further exploring categorical data we looked at our outcome variable "DEP_DEl15" which is a departure delay indicator of 15 Minutes or More (1=Yes). On observing the counts for delayed flights we realized that only ~18% of the flights were classfied as delayed. This highlighted the imabalance in our dataset of the outcome feature. In our ML algorithm implementation we have used methods (Over-representation of minority class) to balance this dataset. 
# MAGIC - Majority (~80%) of the delayed departing flights had delayed previous arriving flights. Chicago, Atlanta, NY and Dallas-FortWorth were the most popular airports with high number of departing and arriving flights.
# MAGIC - The dataset also had ~1.6% cancelled and very few ~0.2% diverted flights. On observing the reasons for cancelled flight from cancellation code, the flights were split acorss various reasons and weather was not the only reason, Also these flights never departed so we could not acurately infer our outcome indicator DEP_DEL15. To be clean and precise we decided to remove the cancelled flights from our dataset. We decided to leave the diverted flights as such because they were very few to begin with and there could be lot of reasons for diversions (including weathe)r which we wanted to account for in our analysis.

# COMMAND ----------

#Frequency tables for categorical data in columns of interest to understand delayed, cancelled and diverted flights
freq_table_year = df_airlines.groupBy("YEAR").count().orderBy('count', ascending=False).show()
freq_table_cancel = df_airlines.groupBy("CANCELLED").count().cache().show()
freq_table_cancel = df_airlines.groupBy("DIVERTED").count().show()
freq_table_delay = df_airlines.groupBy("DEP_DEL15").count().cache().show()
freq_table_arrdelay = df_airlines.groupBy("ARR_DEL15").count().cache().show()
arrival_delay = df_airlines.crosstab('DEP_DEL15','ARR_DEL15').show()
cancel_delay = df_airlines.crosstab('CANCELLED', 'DEP_DEL15').show()
origin_table = df_airlines.groupBy("ORIGIN_CITY_NAME").count().orderBy('count', ascending=False).show()
destination_table = df_airlines.groupBy("DEST_CITY_NAME").count().orderBy('count', ascending=False).show()
cancelreason_table = df_airlines.crosstab("CANCELLED","CANCELLATION_CODE").show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Missing Data
# MAGIC We observed several columns (especially related to diveretd flights) had high number of missing values. We took a high benchmark threshold of 96% and any columns with missing values > 96% of the total data were excluded from the dataset. Below we report the columns that were deleted. The dataset originally had 109 columns and after removing there were 61 columns left.

# COMMAND ----------

#Looking for Missing values
missing_df = df_airlines.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in df_airlines.columns)).toPandas()
missing_df

# COMMAND ----------

#Handling Missing Values: Deleting columns with missing data greater than 96%  
total = df_airlines.count()
threshold = 0.96 * total #threshold to delete
cols_to_drop = []
for col in missing_df.columns:
  val = missing_df[col].values[0]
  if val > threshold:
    cols_to_drop.append(col)
  else:
    pass
print(cols_to_drop) #list of cols to delete
print("total:", total, "threshold to drop:", threshold)

# COMMAND ----------

#drop cols list to be deleted
df_airlines_new = df_airlines.drop(*cols_to_drop) 
column_new = len(df_airlines_new.columns)
print("Number of columns after handling missing data", column_new)
#display(df_airlines_new)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Correlation matrix
# MAGIC To understand multicollinearity in our dataset we built a correlation matrix looking at peasrson correlations for potential columsn of interest that we plan to include in our final model. Looking at the coefficients we see that Quarter and month; Departure delay and Arrival delay; Actual elapsed time and Distance have high correlations (> 0.95).

# COMMAND ----------

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# get non-string columns of interest
df_small = df_airlines_new[['DEP_DEL15','YEAR','QUARTER','MONTH','DAY_OF_WEEK','DAY_OF_MONTH','ORIGIN_AIRPORT_ID','OP_CARRIER_AIRLINE_ID','DEST_AIRPORT_ID','CRS_DEP_TIME','DEP_DELAY','ARR_DELAY','ACTUAL_ELAPSED_TIME','DISTANCE','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY' ]].fillna(0)

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=df_small.columns, outputCol=vector_col)
df_vector = assembler.transform(df_small).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col)

# COMMAND ----------


matrix.collect()[0]["pearson({})".format(vector_col)].values
cor_np = matrix.collect()[0]["pearson({})".format(vector_col)].values
dim = len(cor_np)
cor_mat = cor_np.reshape( (19,19) )
fig, ax = plt.subplots(figsize=(24,24))
sns.heatmap(cor_mat, annot=True, fmt='.2f', cmap = sns.cm.rocket_r)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cancelled Flights
# MAGIC We explained our observations for cancelled flights above and decidied to remove the from the dataset. Below is the filter operation to accomlish this.

# COMMAND ----------

# Unable to classify with accuracy if cancelled flights were delayed due to weather. Few are marked as delayed or not and for majority delayed status is null. We will exclude all the cancelled flights.Diverted flights will be left as such because there could be lot of reasons for diversions and weather could be one of them. We dont want to loose them.

#No. of diverted flights that were cancelled
cancel_time = df_airlines_new.crosstab('DIVERTED', 'CANCELLED').toPandas() 

#Filter out cancelled flights
df_airlines_filter = df_airlines_new.where(df_airlines_new.CANCELLED != 1.0).cache()
#display(df_airlines_filter)

freq_table_cancel = df_airlines_filter.groupBy("CANCELLED").count().show() #confirm no cancelled flights remaining
 

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA to understand trends and patterns in data
# MAGIC Based on our initial EDA and get an idea about which features to explore and engineer we looked at some trends and patterns in our dataset. The headers below describe our highlights from these patterns.

# COMMAND ----------

# create TempView to allow SQL queries
df_airlines.createTempView("airlines")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### _More than 20% of flights delay more than 15 minutes_

# COMMAND ----------

# query departure delay data
delay_hist_df = spark.sql("SELECT dep_delay FROM airlines ORDER BY dep_delay ASC").na.drop().toPandas()
# set-up a new figure with white facecolor
plt.figure(figsize=(10, 5), facecolor='white')
# set-up a new axes
ax = plt.subplot(1, 1, 1)
# set plot appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#C4B6BC')
ax.spines['left'].set_color('#C4B6BC')
ax.tick_params(axis='x', colors='#75686D')
ax.tick_params(axis='y', colors='#75686D')
# plot the histogram
N, bins, patches = ax.hist(x=delay_hist_df.dep_delay.tolist(), cumulative=True, density=True, bins=10000, histtype='bar', data=delay_hist_df)
for i in range(0,345):
    patches[i].set_facecolor('#dedede')
for i in range(345, len(patches)):
    patches[i].set_facecolor('#3D6197')
# place plot title
plt.title(r'Departure delay cumulative distribution', pad=22, fontsize=15, x=0.38, color='#75686D')
# place vertical and horizontal lines
#plt.text(x=2, y=0.99, s='23% of flights delayed more than 15 min', fontsize=15, color='#3D6197', fontweight='bold')
# set y axis params
plt.ylabel('cdf', fontsize=12, labelpad=10, color='#75686D')
plt.yticks(np.arange(0.0, 1.2, step=0.2))
plt.ylim((0.0, 1.0))
# x axis params
plt.xlabel('Delay (in minutes)', fontsize=12, labelpad=10, color='#75686D')
plt.xticks(np.arange(0, 105, step=15))
plt.xlim((0, 90))
# display results
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### _Majority of delays caused by own carrier or late aircrafts_

# COMMAND ----------

# query departure delay data
delay_reasons_df = spark.sql("SELECT SUM(carrier_delay), SUM(late_aircraft_delay), SUM(nas_delay), SUM(weather_delay), SUM(security_delay) FROM airlines").toPandas()
# re-scale values dividing per 1,000
delay_reasons_df = delay_reasons_df/1000
# rename column names
new_col_names = {'sum(carrier_delay)': 'Carrier', 'sum(weather_delay)': 'Weather', 'sum(nas_delay)': 'NAS',
                 'sum(security_delay)': 'Security', 'sum(late_aircraft_delay)': 'Late aircraft'}
delay_reasons_df = delay_reasons_df.rename(new_col_names, axis=1)
# set-up a new figure with white facecolor
plt.figure(figsize=(6, 4), facecolor='white')
# set-up a new axes
ax = plt.subplot(1, 1, 1)
# set plot appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#C4B6BC')
ax.spines['left'].set_color('#C4B6BC')
ax.tick_params(axis='x', colors='#75686D', labelsize=12)
ax.tick_params(axis='y', colors='#75686D')
# plot the barplot
for x in delay_reasons_df.columns:
  if x in ['Carrier', 'Late aircraft']:
    ax.bar(x, delay_reasons_df[x].values, color='#3D6197')
  else:
    ax.bar(x, delay_reasons_df[x].values, color='#dedede')
# place plot title
plt.title(r'Root causes for departure delays', pad=20, fontsize=15, x=0.42, color='#75686D')
# place vertical and horizontal lines
#plt.text(x=-0.4, y=750, s='Majority of delays caused by carrier or late aircrafts', fontsize=15, color='#3D6197', fontweight='bold')
# set y axis params
plt.ylabel('Total delay in 1,000 min', fontsize=12, labelpad=10, color='#75686D')
plt.yticks(np.arange(0.0, 350000, step=50000))
plt.ylim((0.0, 350000))
# display results
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### _Delays are not uniformly distributed over the days_

# COMMAND ----------

# query percentage of delayed flights per day
count_delayed_df = spark.sql("SELECT month, day_of_month, count(op_carrier_fl_num) FROM airlines WHERE dep_del15==1 GROUP BY month, day_of_month, dep_del15 ORDER BY month, day_of_month").toPandas()
count_total_df = spark.sql("SELECT month, day_of_month, count(op_carrier_fl_num) FROM airlines WHERE dep_del15 IS NOT NULL GROUP BY month, day_of_month ORDER BY month, day_of_month").toPandas()
pct_delayed_df = pd.concat([count_delayed_df,count_total_df], axis=1)
pct_delayed_df.columns = ['month', 'day_of_month', 'count_delayed', 'month_2', 'day_of_month_2', 'count_total']
pct_delayed_df['pct_delayed'] = pct_delayed_df['count_delayed']/pct_delayed_df['count_total']
pct_delayed_df['date'] = pct_delayed_df['month'].astype(str) + '/' + pct_delayed_df['day_of_month'].astype(str)
pct_delayed_df = pct_delayed_df.drop(['month', 'day_of_month', 'month_2', 'day_of_month_2', 'count_delayed', 'count_total'], axis=1)
# set-up a new figure with white facecolor
plt.figure(figsize=(15, 6), facecolor='white')
# set-up a new axes
ax = plt.subplot(1, 1, 1)
# set plot appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#C4B6BC')
ax.spines['left'].set_color('#C4B6BC')
ax.tick_params(axis='x', colors='#75686D')
ax.tick_params(axis='y', colors='#75686D')
# plot the lineplot
ax.plot(pct_delayed_df['date'], pct_delayed_df['pct_delayed'], color='#3D6197', linestyle='-', linewidth=2)
# place plot title
plt.title(r'Departure delay incidence over time', pad=22, fontsize=15, x=0.36, color='#75686D')
# place vertical and horizontal lines
#plt.text(x=5, y=0.6, s='Delays tend to concentrate on specific days', fontsize=15, color='#3D6197', fontweight='bold')
# set y axis params
plt.ylabel('Pct of flights with delays >15min', fontsize=12, labelpad=9, color='#75686D')
plt.yticks(np.arange(0.0, 0.7, step=0.1))
plt.ylim((0.0, 0.6))
# x axis params
every_nth = 20
for n, label in enumerate(ax.xaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)
ax.tick_params(axis='both', which='both', length=0)
# display results
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### _Delays are not uniformly distributed among the airlines_

# COMMAND ----------

# query percentage of delayed flights per airline
count_delayed_by_airline_df = spark.sql("SELECT op_carrier_airline_id, count(op_carrier_fl_num) AS count_delayed FROM airlines WHERE dep_del15==1 GROUP BY op_carrier_airline_id ORDER BY count(op_carrier_fl_num) DESC").toPandas()
count_total_by_airline_df = spark.sql("SELECT op_carrier_airline_id, count(op_carrier_fl_num) AS count_total FROM airlines WHERE dep_del15 IS NOT NULL GROUP BY op_carrier_airline_id ORDER BY count(op_carrier_fl_num) DESC").toPandas()
count_delayed_by_airline_df.set_index('op_carrier_airline_id', inplace=True)
count_total_by_airline_df.set_index('op_carrier_airline_id', inplace=True)
pct_delayed_by_airline_df = count_delayed_by_airline_df.join(count_total_by_airline_df)
pct_delayed_by_airline_df = pct_delayed_by_airline_df[pct_delayed_by_airline_df['count_total']>1000]
pct_delayed_by_airline_df['pct_delayed'] = pct_delayed_by_airline_df['count_delayed']/pct_delayed_by_airline_df['count_total']
pct_delayed_by_airline_df = pct_delayed_by_airline_df.drop(['count_delayed', 'count_total'], axis=1)
pct_delayed_by_airline_df.sort_values(by='pct_delayed', axis=0, ascending=False, inplace=True)
# set-up a new figure with white facecolor
plt.figure(figsize=(15, 6), facecolor='white')
# set-up a new axes
ax = plt.subplot(1, 1, 1)
# set plot appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#C4B6BC')
ax.spines['left'].set_color('#C4B6BC')
ax.tick_params(axis='x', colors='#75686D', labelsize=12)
ax.tick_params(axis='y', colors='#75686D')
# plot the barplot
for x in pct_delayed_by_airline_df.index:
  ax.bar(str(x), pct_delayed_by_airline_df.loc[x, 'pct_delayed'], color='#3D6197')
# place plot title
plt.title(r'Delay incidence per airline', pad=22, fontsize=15, x=0.27, color='#75686D')
# place vertical and horizontal lines
#plt.text(x=-0.4, y=0.48, s='Delays tend to concentrate on specific airlines', fontsize=15, color='#3D6197', fontweight='bold')
# set y axis params
plt.ylabel('Pct of flights with delays >15min', fontsize=12, labelpad=9, color='#75686D')
plt.yticks(np.arange(0.0, 0.6, step=0.1))
plt.ylim((0.0, 0.5))
# set x axis params
plt.xlabel('Airline ID', fontsize=12, labelpad=12, color='#75686D')
# display results
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### _Delays are not uniformly distributed among the origin airports_

# COMMAND ----------

# query percentage of delayed flights per airport
count_delayed_by_airport_df = spark.sql("SELECT origin, count(op_carrier_fl_num) AS count_delayed FROM airlines WHERE dep_del15==1 GROUP BY origin ORDER BY count_delayed DESC").toPandas()
count_total_by_airport_df = spark.sql("SELECT origin, count(op_carrier_fl_num) AS count_total FROM airlines WHERE dep_del15 IS NOT NULL GROUP BY origin ORDER BY count_total DESC").toPandas()
count_delayed_by_airport_df.set_index('origin', inplace=True)
count_total_by_airport_df.set_index('origin', inplace=True)
pct_delayed_by_airport_df = count_delayed_by_airport_df.join(count_total_by_airport_df)
pct_delayed_by_airport_df['pct_delayed'] = pct_delayed_by_airport_df['count_delayed']/pct_delayed_by_airport_df['count_total']
pct_delayed_by_airport_df = pct_delayed_by_airport_df.drop(['count_delayed', 'count_total'], axis=1)
pct_delayed_by_airport_df.sort_values(by='pct_delayed', axis=0, ascending=False, inplace=True)
# set-up a new figure with white facecolor
plt.figure(figsize=(15, 6), facecolor='white')
# set-up a new axes
ax = plt.subplot(1, 1, 1)
# set plot appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#C4B6BC')
ax.spines['left'].set_color('#C4B6BC')
ax.tick_params(axis='x', colors='#75686D', labelsize=12)
ax.tick_params(axis='y', colors='#75686D')
# plot the barplot
for x in pct_delayed_by_airport_df.index:
  ax.bar(str(x), pct_delayed_by_airport_df.loc[x, 'pct_delayed'], color='#3D6197')
# place plot title
plt.title(r'Delay incidence per origin airport', pad=22, fontsize=15, x=0.43, color='#75686D')
# place vertical and horizontal lines
#plt.text(x=-0.4, y=0.48, s='Delays tend to concentrate on specific origin airports', fontsize=15, color='#3D6197', fontweight='bold')
# set y axis params
plt.ylabel('Pct of flights with delays >15min', fontsize=12, labelpad=9, color='#75686D')
plt.yticks(np.arange(0.0, 0.6, step=0.1))
plt.ylim((0.0, 0.5))
# set x axis params
every_nth = 20
for n, label in enumerate(ax.xaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)
ax.tick_params(axis='both', which='both', length=0)
plt.xlabel('Origin airport', fontsize=12, labelpad=10, color='#75686D')
# display results
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### _Delays are not uniformly distributed among the destination airports_

# COMMAND ----------

# query percentage of delayed flights per airport
count_delayed_by_dest_airport_df = spark.sql("SELECT dest, count(op_carrier_fl_num) AS count_delayed FROM airlines WHERE dep_del15==1 GROUP BY dest ORDER BY count_delayed DESC").toPandas()
count_total_by_dest_airport_df = spark.sql("SELECT dest, count(op_carrier_fl_num) AS count_total FROM airlines WHERE dep_del15 IS NOT NULL GROUP BY dest ORDER BY count_total DESC").toPandas()
count_delayed_by_dest_airport_df.set_index('dest', inplace=True)
count_total_by_dest_airport_df.set_index('dest', inplace=True)
pct_delayed_by_dest_airport_df = count_delayed_by_dest_airport_df.join(count_total_by_dest_airport_df)
pct_delayed_by_dest_airport_df['pct_delayed'] = pct_delayed_by_dest_airport_df['count_delayed']/pct_delayed_by_dest_airport_df['count_total']
pct_delayed_by_dest_airport_df = pct_delayed_by_dest_airport_df.drop(['count_delayed', 'count_total'], axis=1)
pct_delayed_by_dest_airport_df.sort_values(by='pct_delayed', axis=0, ascending=False, inplace=True)
# set-up a new figure with white facecolor
plt.figure(figsize=(15, 6), facecolor='white')
# set-up a new axes
ax = plt.subplot(1, 1, 1)
# set plot appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#C4B6BC')
ax.spines['left'].set_color('#C4B6BC')
ax.tick_params(axis='x', colors='#75686D', labelsize=12)
ax.tick_params(axis='y', colors='#75686D')
# plot the barplot
for x in pct_delayed_by_dest_airport_df.index:
  ax.bar(str(x), pct_delayed_by_dest_airport_df.loc[x, 'pct_delayed'], color='#3D6197')
# place plot title
plt.title(r'Delay incidence per destination airport', pad=22, fontsize=15, x=0.37, color='#75686D')
# place vertical and horizontal lines
#plt.text(x=5, y=0.49, s='Delays tend to concentrate on specific destination airports', fontsize=15, color='#3D6197', fontweight='bold')
# set y axis params
plt.ylabel('Pct of flights with delays >15min', fontsize=12, labelpad=9, color='#75686D')
plt.yticks(np.arange(0.0, 0.6, step=0.1))
plt.ylim((0.0, 0.5))
# set x axis params
every_nth = 20
for n, label in enumerate(ax.xaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)
ax.tick_params(axis='both', which='both', length=0)
plt.xlabel('Destination airport', fontsize=12, labelpad=10, color='#75686D')
#plt.xticks([])
# display results
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Based on our observations in the dataset on EDA, main reasons for flight delay in US as reported by the Bureau of Transportation Statistics [7] and fascilitate join with weather dataset, we got some hints about features to explore and engineer. The following features were engineered and added to the full dataset:
# MAGIC 
# MAGIC - Timestamp reflecting airline scheduled departure date and time: We created a timestamp in utc and unix format by concatenating flight date and CRS_DEP_TIME to fasciliate join with the weather data. Conversion to utc was done as weather data had time in utc format. However when we joined the two datasets together we decided to convert to unix as the join was significantly faster using unix timestamps.
# MAGIC 
# MAGIC - Timestamp 2 hours prior to scheduled departure time: This was added to the dataset as we will are building models that predict flight delays 2 hours prior to scheduled departure time. This will be regarded as our prediction time and used for further feature engineering related to time of departure.
# MAGIC 
# MAGIC - Average delayed flights (rolling window 30 days) for every aircraft: was added to account for an aircraft's condition (aging, chronic faults etc.) on flight delays. Unique aircrafts were identified by tail_num.
# MAGIC 
# MAGIC - Frequency of delays in the departure airport (for all airlines) in the past 2, 4, 8 and 12 hours: to account for local issues in the departing airport may cause a series of flights from different airlines to delay (e.g. weather conditions, security incidents, etc.)
# MAGIC 
# MAGIC - Frequency of delays in the destination airport (for all airlines) in the past 2, 4, 8 and 12 hours: to account for local issues in the destination airport may cause a series of flights from different airlines to delay in the origin (e.g. weather conditions, security incidents, etc.)
# MAGIC 
# MAGIC - Frequency of delays in the most important hubs (for all airlines) in the past 2, 4, 8 and 12 hours: to account for local issues in important hubs of the system may cause a series of flight from different airlines in different airports to delay
# MAGIC 
# MAGIC - Frequency of delays of the same airline (in the departure airport) in the past 2, 4, 8 and 12 hours: to account for majority of delays are classified as under the air carrier control, so a specific carrier previous delays might be a good indicative of operational problems that might propagate
# MAGIC 
# MAGIC - Frequency of delays of the same airline (in the most important hubs) in the past 2, 4, 8 and 12 hours: to account for operational problems in airport hubs which might propagate faster to the system than isolated problems. We will describe how we identified hubs in the relevant section below.
# MAGIC 
# MAGIC - Frequency of late arrivals in the departure airport (for all airlines) in the past 2, 4, 8 and 12 hours: late arriving aircrafts are the second most prevalent cause for flight departure delays
# MAGIC 
# MAGIC - Percentage of flights delayed due to weather every hour: to enable further feature engineering for weather related features we calculated and included % of flights delayed due to weather every hour.
# MAGIC 
# MAGIC - Part of the day (Morning, Afternoon etc.): to account for the snowball effect of flights getting delayed earlier in the day on flights on flights departing later in the day.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setting pipeline to create features

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Creating timestamp reflecting airline scehduled departure date and time

# COMMAND ----------

#Create local Timestamp column by joining Flight date and Departure time
df_airlines_filter = df_airlines_filter.withColumn('timestamp', 
                    sf.concat(sf.col('FL_DATE'),sf.lit(' '), sf.col('CRS_DEP_TIME')))
#display(df_airlines_filter)

# COMMAND ----------

# MAGIC %md
# MAGIC User defined function used in feature engineering and creation

# COMMAND ----------

#UDF`s for feature engineering

## UDF for getting timezone from city
def time_zone(city):
  # initialize Nominatim API
  geolocator = Nominatim(user_agent="geoapiExercises")
  
  if "/" in city: #take first city of airports serving MSA`s 
    city=city.split("/")[0]

  #get long and latitude
  location = geolocator.geocode(city) #location.latitude and location.longitude
  #print("city:",city)
  #print("lngitude:", location.longitude)

  # pass the Latitude and Longitud into a timezone_at and return timezone
  obj = TimezoneFinder()
  result = obj.timezone_at(lng=location.longitude, lat=location.latitude)
  return result


## UDF Convert local time timestamp to utc timestamp using timezones
def utc_converter(timezone, time):
  local = pytz.timezone(timezone)
  #naive = datetime.strptime("2015-02-10 1729", "%Y-%m-%d %H%M")
  if time != "":
    naive = datetime.strptime(time, "%Y-%m-%d %H%M")
    local_dt = local.localize(naive, is_dst=None)
    utc_dt = local_dt.astimezone(pytz.utc)
  else:
    utc_dt = "Null"
  return utc_dt

def utc_converter2(timezone, time):
  try:
    local = pytz.timezone(timezone)
    naive = datetime.strptime(time, "%Y-%m-%d %H%M")
    local_dt = local.localize(naive, is_dst=None)
    utc_dt = local_dt.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%S")
  except:
    return None
  return utc_dt

#UDF for port hub status
def hub_port(rank): 
  if rank <= 15: #took top 15 airports with maximum total(arriving and departing flights) as hubs
    port = "hub"
  else:
    port = "non-hub"
  return port

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Identify airports as hubs
# MAGIC In order to label airports as hubs we calculated the total number of arriving and departing flights from every airport and then ranked them based on the count. The top 15 airports were classified as hubs and rest as non-hubs and this was added as a column.

# COMMAND ----------

from pyspark.sql.functions import col
tab_origin = df_airlines.groupBy("ORIGIN_AIRPORT_ID").count().withColumnRenamed("ORIGIN_AIRPORT_ID", "AIRPORT_ID") #total flights from origin
display(tab_origin)
tab_dest = df_airlines.groupBy("DEST_AIRPORT_ID").count().withColumnRenamed("DEST_AIRPORT_ID", "AIRPORT_ID") #total flights from dest
display(tab_dest)
result = tab_origin.unionByName(tab_dest) #union of both
display(result)
result2 = result.groupBy("AIRPORT_ID").sum('count').orderBy('sum(count)', ascending=False) #total sum every airport
display(result2)
windowSpec  = Window.partitionBy().orderBy(col("sum(count)").desc()) 
result3 = result2.withColumn("airport_rank",rank().over(windowSpec)) #ranking
display(result3)

#using udf to main dataframe
HubUDF = udf(lambda z: hub_port(z),StringType())
result4= result3.withColumn("hub_status", HubUDF(sf.col("airport_rank")))

#result4.withColumn("json", sf.create_map(["AIRPORT", "dma"])).show(truncate=False)
hub_lookup = {row['AIRPORT_ID']:row['hub_status'] for row in result4.collect()}

# Applying hub lookup UDF on spark dataframe to create port_hub_status column
hub_mainUDF = udf(lambda z: hub_lookup[z], returnType= StringType())

#using lookup table to update main dataframe
df_airlines_filter = df_airlines_filter.withColumn("origin_hub_status", hub_mainUDF(sf.col("ORIGIN_AIRPORT_ID")))
df_airlines_filter = df_airlines_filter.withColumn("dest_hub_status", hub_mainUDF(sf.col("DEST_AIRPORT_ID")))
#display(df_airlines_filter)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Converting timestamp into utc and unix format
# MAGIC We had to addtional engineering as timestamp we created were in local times in the airlines datset. We used the geolocator 'nominatim' API to add time zones to every airport and then used it to get the utc and unix times.

# COMMAND ----------

#split city,state into city
from pyspark.sql.functions import split
from pyspark.sql.functions import col,sum
df_airlines_filter = df_airlines_filter.withColumn("ORG_CITY", split(col("ORIGIN_CITY_NAME"), ",").getItem(0))

# COMMAND ----------

uniq_cities = df_airlines_filter.agg(sf.collect_set('ORG_CITY')).first() # Creating static Lookup table for cities
city_to_timezone_lookup = {city: time_zone(city) for city in uniq_cities[0]} #static lookup table

# COMMAND ----------

# Applying city_to_timezone_lookup UDF on spark dataframe to create timezone column
TimezoneUDF = udf(lambda z: city_to_timezone_lookup[z], returnType= StringType())

# COMMAND ----------

df_airlines_filter = df_airlines_filter.withColumn("time_zone", TimezoneUDF(sf.col("ORG_CITY")))
display(df_airlines_filter)

# COMMAND ----------

#Applying utc_converter on spark dataframe to create utc_timestamp column
utf_UDF = udf(utc_converter2) #TimestampType())
df_airlines_filter = df_airlines_filter.withColumn('utc_timestamp', utf_UDF('time_zone','timestamp'))

# COMMAND ----------

#convert utc timestamp to unix timestamp
df_airlines_filter = df_airlines_filter.withColumn('depart_unix_timestamp', sf.unix_timestamp(sf.col('utc_timestamp'), "yyyy-MM-dd'T'HH:mm:ss"))
#display(df_airlines_filter)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Adding prediction time feature
# MAGIC This is the time that was used to create further features related to average delays an frequency of delays for all airports and airlines.

# COMMAND ----------

#Calculating unix time 2 hrs prior to departure time stamp for feature engineering
df_airlines_filter = df_airlines_filter.withColumn("depart_unix_prior2hr", sf.col("depart_unix_timestamp")- 7200)
display(df_airlines_filter)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Average delayed flights (rolling window 30 days) for every aircraft 

# COMMAND ----------

#Average delayed flights per aircraft
from pyspark.sql.window import Window
#function to calculate number of seconds from number of days
days = lambda i: i * 86400
w = Window().partitionBy(sf.col("TAIL_NUM")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-days(30), 0)
airlines_rolling = df_airlines_filter.withColumn('rolling_average', sf.avg("DEP_DEL15").over(w))
display(airlines_rolling)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Frequency of delayed flights at origin and destination airport for all airlines in the past 2, 4, 8 and 12 hours

# COMMAND ----------

#Frequency for delayed flights at origin airport for all airlines
from pyspark.sql.window import Window
#function to calculate number of seconds from hours
hours = lambda i: i * 3600
w = Window().partitionBy(sf.col("ORIGIN")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(2), 0)
airlines_rolling_originairport_2 = airlines_rolling.withColumn('delay_2hrs_originport', sf.sum("DEP_DEL15").over(w))
w = Window().partitionBy(sf.col("ORIGIN")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(4), 0)
airlines_rolling_originairport_4 = airlines_rolling_originairport_2.withColumn('delay_4hrs_originport', sf.sum("DEP_DEL15").over(w))
w = Window().partitionBy(sf.col("ORIGIN")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(8), 0)
airlines_rolling_originairport_8 = airlines_rolling_originairport_4.withColumn('delay_8hrs_originport', sf.sum("DEP_DEL15").over(w))
w = Window().partitionBy(sf.col("ORIGIN")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(12), 0)
airlines_rolling_originairport_12 = airlines_rolling_originairport_8.withColumn('delay_12hrs_originport', sf.sum("DEP_DEL15").over(w))
display(airlines_rolling_originairport_12)

# COMMAND ----------

#Rolling frequency of delayed flights at destination airport
#function to calculate number of seconds from hours
hours = lambda i: i * 3600
w = Window().partitionBy(sf.col("DEST")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(2), 0)
airlines_rolling_destairport_2 = airlines_rolling_originairport_12.withColumn('delay_2hrs_destport', sf.sum("DEP_DEL15").over(w))
w = Window().partitionBy(sf.col("DEST")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(4), 0)
airlines_rolling_destairport_4 = airlines_rolling_destairport_2.withColumn('delay_4hrs_destport', sf.sum("DEP_DEL15").over(w))
w = Window().partitionBy(sf.col("DEST")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(8), 0)
airlines_rolling_destairport_8 = airlines_rolling_destairport_4.withColumn('delay_8hrs_destport', sf.sum("DEP_DEL15").over(w))
w = Window().partitionBy(sf.col("DEST")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(12), 0)
airlines_rolling_destairport_12 = airlines_rolling_destairport_8.withColumn('delay_12hrs_destport', sf.sum("DEP_DEL15").over(w))
display(airlines_rolling_destairport_12)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Frequency of delayed flights by airlines in the past 2, 4, 8 and 12 hours

# COMMAND ----------

#frequency for delayed flights by airlines at departure airport
#function to calculate number of seconds from hours
hours = lambda i: i * 3600
w = Window().partitionBy(sf.col("OP_CARRIER_AIRLINE_ID")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(2), 0)
airlines_rolling_originairline_2 = airlines_rolling_destairport_12.withColumn('delay_2hrs_orgairline', sf.sum("DEP_DEL15").over(w))
w = Window().partitionBy(sf.col("OP_CARRIER_AIRLINE_ID")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(4), 0)
airlines_rolling_originairline_4 = airlines_rolling_originairline_2.withColumn('delay_4hrs_orgairline', sf.sum("DEP_DEL15").over(w))
w = Window().partitionBy(sf.col("OP_CARRIER_AIRLINE_ID")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(8), 0)
airlines_rolling_originairline_8 = airlines_rolling_originairline_4.withColumn('delay_8hrs_orgairline', sf.sum("DEP_DEL15").over(w))
w = Window().partitionBy(sf.col("OP_CARRIER_AIRLINE_ID")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(12), 0)
airlines_rolling_originairline_12 = airlines_rolling_originairline_8.withColumn('delay_12hrs_orgairline', sf.sum("DEP_DEL15").over(w))
display(airlines_rolling_originairline_12)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Frequency of late arrivals in the departure airport (for all airlines) in the past 2, 4, 8 and 12 hours

# COMMAND ----------

#Frequency of late arrivals in the departure airport (for all airlines) in the past 2, 4, 8 and 12 hours
#function to calculate number of seconds from hours
hours = lambda i: i * 3600
w = Window().partitionBy(sf.col("ORIGIN")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(2), 0)
airlines_arrdelay_originairport_2 = airlines_rolling_originairline_12.withColumn('arrdelay_2hrs_originport', sf.sum("ARR_DEL15").over(w))
w = Window().partitionBy(sf.col("ORIGIN")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(4), 0)
airlines_arrdelay_originairport_4 = airlines_arrdelay_originairport_2.withColumn('arrdelay_4hrs_originport', sf.sum("ARR_DEL15").over(w))
w = Window().partitionBy(sf.col("ORIGIN")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(8), 0)
airlines_arrdelay_originairport_8 = airlines_arrdelay_originairport_4.withColumn('arrdelay_8hrs_originport', sf.sum("ARR_DEL15").over(w))
w = Window().partitionBy(sf.col("ORIGIN")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(12), 0)
airlines_arrdelay_originairport_12 = airlines_arrdelay_originairport_8.withColumn('arrdelay_12hrs_originport', sf.sum("ARR_DEL15").over(w))
display(airlines_arrdelay_originairport_12)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Frequency of delays of airlines in the most important hubs in the past 2, 4, 8 and 12 hours

# COMMAND ----------

#Frequency of delays of the same airline (in the most important hubs) in the past 2, 4, 8 and 12 hours
#function to calculate number of seconds from hours
hours = lambda i: i * 3600
w = Window().partitionBy(sf.col("origin_hub_status")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(2), 0)
airlines_delay_huborigin_2 = airlines_arrdelay_originairport_12.withColumn('delay_2hrs_originhub', sf.sum("DEP_DEL15").over(w))
w = Window().partitionBy(sf.col("origin_hub_status")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(4), 0)
airlines_delay_huborigin_4 = airlines_delay_huborigin_2.withColumn('delay_4hrs_originhub', sf.sum("DEP_DEL15").over(w))
w = Window().partitionBy(sf.col("origin_hub_status")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(8), 0)
airlines_delay_huborigin_8 = airlines_delay_huborigin_4.withColumn('delay_8hrs_originhub', sf.sum("DEP_DEL15").over(w))
w = Window().partitionBy(sf.col("origin_hub_status")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(12), 0)
airlines_delay_huborigin_12 = airlines_delay_huborigin_8.withColumn('delay_12hrs_originhub', sf.sum("DEP_DEL15").over(w))
display(airlines_delay_huborigin_12)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Percentage of flights delayed due to weather every hour
# MAGIC 
# MAGIC The weather_delay feature gives the number of minutes a flight was delayed due to weather. However no identifier was available to indicate  if a flight was delayed due to weather. We created an indicator feature for this. If a flight was delayed (as there were some flights which had weather delay of > 0 minutes but were not overall delayed) and weather delay >0 minutes (regardless of other delay reasons), the flight was labelled as delayed due to weather. This feature was used to calculate the percentage of flights delayed due to weather.

# COMMAND ----------

#UDF to Indicate delayed flight due to weather
def weatherdelay_indicate(delayed,minutes):
  if minutes != None:
    if delayed == 1 and minutes > 0:
      weather_delay = 1
      print("minutes:", minutes)
    else:
      weather_delay = 0
  else:
    weather_delay = 0
  return weather_delay
wdelayUDF = udf(lambda z,y: weatherdelay_indicate(z,y),IntegerType())

# COMMAND ----------

df_airlines_final2= airlines_delay_huborigin_12.withColumn("weath_delay", wdelayUDF(sf.col("DEP_DEL15"),("WEATHER_DELAY")))
display(df_airlines_final2) 

# COMMAND ----------

#confriming % of flights delayed due to weather based on our definition
wdelay_table = df_airlines_final2.groupBy("weath_delay").count().show()
weath_delay = df_airlines_final2.crosstab('DEP_DEL15', 'weath_delay').show()
weath_delay

# COMMAND ----------

hours = lambda i: i * 3600
w = Window().partitionBy(sf.col("ORIGIN")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(1), 0)
df_airlines_final3 = df_airlines_final2.withColumn('total_flights_perhr', sf.count("ORIGIN_AIRPORT_ID").over(w))
#display(df_airlines_final3)

# COMMAND ----------

w = Window().partitionBy(sf.col("ORIGIN")).orderBy(sf.col("depart_unix_prior2hr")).rangeBetween(-hours(1), 0)
df_airlines_final4 = df_airlines_final3.withColumn('total_w_delay', sf.sum("weath_delay").over(w))
#display(df_airlines_final4)

# COMMAND ----------

df_airlines_final5 = df_airlines_final4.withColumn('percent_wdelay',(df_airlines_final4.total_w_delay/df_airlines_final4.total_flights_perhr)*100)
display(df_airlines_final5) 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Checkpoint 
# MAGIC 
# MAGIC To reduce running time given how big the dataset is we created a checkpoint, pushed interim dataset to the blob and then pulled from the blob to create further features.

# COMMAND ----------

#Write interim dataframe to the blob
df_airlines_final5.write.parquet(f"{blob_url}/df_airlines_final_full_v1")

# Load interim flights data from the blob
df_airlines_finalb = spark.read.parquet(f"{blob_url}/df_airlines_final_full_v1")
display(df_airlines_finalb)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Testing correct data counts from Blob
# MAGIC Our team had some issue with writing ot the blob and our data was reduced while writing to the blob. We tested data from this checkpoint ot make sure we did not run into any issues. 

# COMMAND ----------

#test data from blob
df_airlines_finalb.count()

# COMMAND ----------

#Check Number of flights delayed due to weather is same as before pushing to blob - Testing
wdelay_table = df_airlines_finaltest.groupBy("weath_delay").count().show()
weath_delay = df_airlines_finaltest.crosstab('DEP_DEL15', 'weath_delay').show()
weath_delay

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Feature reflecting part of the day (Morning etc.)

# COMMAND ----------

# Adding part of the day feature
import datetime
from datetime import *

#UDf to extract hour from hh:mm format
def hour_convert(time):
  try:
    time_split = datetime.strptime(str(time), "%H%M")
    hr = time_split.hour
  except:
    hr = int(time)
  return hr

#UDF to classify part of the day using hour
def part_of_day(hr):
  day_part = 0
  if hr >= 5 and hr < 12:
    day_part = "Morning"
  elif hr >= 12 and hr < 17:
    day_part = "Afteroon"
  elif hr >= 17 and hr < 21:
    day_part = "Evening"
  else:
    day_part = "Night"
  return day_part


# COMMAND ----------

#Apply hour and day part UDF to dataframe
HourUDF = udf(lambda z: hour_convert(z),IntegerType())
partUDF = udf(lambda z: part_of_day(z), StringType())
df_airlines_finalb1= df_airlines_finalb.withColumn("DEP_HOUR", HourUDF(sf.col("CRS_DEP_TIME")))
display(df_airlines_finalb1)

# COMMAND ----------

df_airlines_finalb2 = df_airlines_finalb1.withColumn("Part_of_Day", partUDF(sf.col("DEP_HOUR")))
display(df_airlines_finalb2)

# COMMAND ----------

#Confirm part of the day and hours match
hr_table = df_airlines_finalb2.groupBy("DEP_HOUR").count().show()
hr_part = df_airlines_finalb2.crosstab('DEP_HOUR', 'PART_of_Day').show()
hr_part

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Dataset 

# COMMAND ----------

#Final DataFrame
# dataframe.write.fileformat(f"{blob_url}")
df_airlines_finalb2.write.parquet(f"{blob_url}/df_airlines_final_full_v7")

# COMMAND ----------

#confirm if pushed to the blob
display(dbutils.fs.ls(blob_url))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Help links and References
# MAGIC 
# MAGIC Operations (aggregation, cumulative sum, rolling windows):
# MAGIC - https://excelkingdom.blogspot.com/2017/12/how-to-calculate-cumulative-sum-or_15.html
# MAGIC - https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/968100988546031/157591980591166/8836542754149149/latest.html
# MAGIC - https://stackoverflow.com/questions/45806194/pyspark-rolling-average-using-timeseries-data
# MAGIC - https://www.analyticsvidhya.com/blog/2019/12/6-powerful-feature-engineering-techniques-time-series/
# MAGIC - https://stackoverflow.com/questions/45946349/python-spark-cumulative-sum-by-group-using-dataframe
# MAGIC - https://stackoverflow.com/questions/60869614/pyspark-how-to-extract-hour-from-timestamp 
# MAGIC 
# MAGIC 
# MAGIC Datetime conversion:
# MAGIC - https://www.kite.com/python/answers/how-to-convert-local-datetime-to-utc-in-python 
# MAGIC 
# MAGIC UDF's in PySpark:
# MAGIC - https://sparkbyexamples.com/pyspark/pyspark-udf-user-defined-function/
# MAGIC 
# MAGIC Syllabus link:
# MAGIC - https://docs.google.com/document/d/1BTbc6znZe3wTpdIVeun66K8AkzUlfYW1dlDPZkbZ5NM/edit
# MAGIC 
# MAGIC Project doc:
# MAGIC - https://docs.google.com/document/d/1eViHlmcDUCqse482TE8ReFth7koQa4C1wLP-EBqKRGY/edit
# MAGIC 
# MAGIC EDA:
# MAGIC - https://towardsdatascience.com/a-practical-guide-for-exploratory-data-analysis-flight-delays-f8a713ef7121 
# MAGIC 
# MAGIC Columns of Interest: 
# MAGIC - FL_DATE, OP_UNIQUE_CARRIER, OP_CARRIER_AIRLINE_ID, TAIL_NUM, ORIGIN, ORIGIN_AIRPORT_ID, ORIGIN_CITY_NAME, DEST_AIRPORT_ID, DEST, DEST_CITY_NAME, DEP_TIME, DEP_DELAY, DEP_DEL15, ARR_TIME, ARR_DELAY, ARR_DEL15, CANCELLED, CANCELLATION_CODE, DIVERTED, AIR_TIME, FLIGHTS, DISTANCE, CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY, DIVE_AIRPORT_LANDINGS
# MAGIC 
# MAGIC DEPENDENT VARIABLE:
# MAGIC - DEP_DEL15 (1- IF flight is 15 min delayed, 0 otherwise)
# MAGIC 
# MAGIC Dataset Links:
# MAGIC - Link: https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGK
# MAGIC - Link to data dictionary: https://www.transtats.bts.gov/Glossary.asp?index=C 
# MAGIC - Flights data helpful link: https://www.transtats.bts.gov/homepage.asp 
