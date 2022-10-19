# Databricks notebook source
# MAGIC %md
# MAGIC # Weather Dataset Exploration
# MAGIC 
# MAGIC <a href='$./fp_main_notebook_final'>To return to main notebook click here</a>.
# MAGIC 
# MAGIC #### Starter Code (import libraries, etc)

# COMMAND ----------

from pyspark.sql.functions import col,sum
from pyspark.sql.functions import isnan, when, count
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
try:
  from pyspark_dist_explore import hist
except:
  !pip install pyspark_dist_explore
  from pyspark_dist_explore import hist

import re
import ast
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

# Init script to create the blob URL
# Put this at the top of every notebook
from pyspark.sql.functions import *

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

# Load 2015 Q1 for Flights
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")
# Load the 2015 Q1 for Weather
df_weather_3m = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(col('DATE') < "2015-04-01T00:00:00.000")
# Load entire Weather Dataset
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*")
df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")

weather_data_trimmed_full_USonly2 = spark.read.parquet(f"{blob_url}/weather_data_trimmed_full_USonly2/*")
weather_data_trimmed_full_USonly2_3m = spark.read.parquet(f"{blob_url}/weather_data_trimmed_full_USonly2/*").filter(col('DATE') < "1427846400")
# df_weather_trimmed = spark.read.parquet(f"{blob_url}/weather_data_trimmed/*")
# df_weather_trimmed2 = spark.read.parquet(f"{blob_url}/weather_data_trimmed2/*")
# df_weather_trimmed3 = spark.read.parquet(f"{blob_url}/weather_data_trimmed3/*")

joined_data_all_v1 = spark.read.parquet(f"{blob_url}/joined_data_all_v1/*")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Analysis of Joined Data
# MAGIC 
# MAGIC The main purpose of this section was to reconcile the bad_weather_prediction feature that we created to the actual dataset (flight delays caused by weather) to see how accurate the predictions were. However, as we have noted extensively in the main notebook, the unanticpated dropping of over 90% of the airlines & weather datasets that occurred during the join() process caused this analysis to be inconclusive.

# COMMAND ----------

# Reconcile bad_weather_prediction with actual flight delays that say they were caused by weather.
print(joined_data_all_v1.count())
# display(joined_data_all_v1)

# Use Kanikas indicator variable
bad_weather_answers = joined_data_all_v1.filter((col('weath_delay') > 0))
bad_weather_answers.summary("count", "mean", "stddev", "min", "10%", "25%", "50%", "75%", "90%", "max").toPandas()

# COMMAND ----------

# Reconcile bad_weather_prediction with actual flight delays that say they were caused by weather.
print(joined_data_all_v1.corr('weath_delay', 'Bad_Weather_Prediction'))

# COMMAND ----------

columns_of_int = ['WND_direction_angle','WND_speed','CIG_ceiling_height_dimension','VIS_distance_dimension','TMP_air_temperature', 'DEW_dew_point_temperature', 'SLP_sea_level_pressure']
for c in columns_of_int:
  fig, ax = plt.subplots()
  hist(ax, bad_weather_answers.select(c), bins = 50, color=['red'])
  ax.set_xlabel(c)
  ax.set_ylabel('Frequency')
  ax.set_title('Actual Bad Weather in Joined Data: Checking for Sig Deviation from Predicted')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Analysis of the "Checkpointed Data" (Cleaned Weather Data that we have saved into the blob)

# COMMAND ----------

# MAGIC %md
# MAGIC We perform a quick check of the checkpointed data to see if there are indeed 42 million rows (as saved), and if there are any significant blank or missing data/features. They have all been accounted for - there are a couple of missing Station and Name labels, but since they represent <1% of the 42 million rows it is insignificant.

# COMMAND ----------

print(weather_data_trimmed_full_USonly2.count())
display(weather_data_trimmed_full_USonly2)

# COMMAND ----------

# Checking for no NAs in the rows we are interested in.
weather_nas = weather_data_trimmed_full_USonly2.select([count(when(col(c).isNull(), c)).alias(c) for c in weather_data_trimmed_full_USonly2.columns])
display(weather_nas)

# COMMAND ----------

# Checking for no Blanks in the rows we are interested in.
weather_blanks = weather_data_trimmed_full_USonly2.select([count(when((col(c) == '' ), c)).alias(c) for c in weather_data_trimmed_full_USonly2.columns])
# df_weather.select([count(when(col(c).contains('None') | \
#                             col(c).contains('NULL') | \
#                              | \
#                             col(c).isNull() | \
#                             isnan(c), c 
#                            )).alias(c)
#                     for c in df_weather.columns])
display(weather_blanks)

# COMMAND ----------

# MAGIC %md
# MAGIC We then plot a histogram and show tables sorted by percentage occurance as an EDA to see if there are commonalities or anything unusual within each of the features below.
# MAGIC 
# MAGIC First, we see if wind angles prove to be significant in terms of involvement in adverse weather conditions. The even distribution here does not give us any meaningful conclusions.

# COMMAND ----------

weather_data_trimmed_full_USonly2.createOrReplaceTempView("weather_data_trimmed_full_USonly2")
tot = weather_data_trimmed_full_USonly2.count()
i = 'WND_direction_angle'
table_equivalent = sqlContext.sql(f"SELECT {i}, count({i}), bround(count({i})/{tot} * 100, 2) AS Percentage FROM weather_data_trimmed_full_USonly2 GROUP BY {i} ORDER BY count({i}) DESC")
display(table_equivalent)

# for c in weird_columns:
fig, ax = plt.subplots()
hist(ax, weather_data_trimmed_full_USonly2.select(i), bins = 50, color=['red'])
ax.set_xlabel('Wind Angle (in Degrees)')
ax.set_ylabel('Frequency')
ax.set_title('Even/Semi-Normal Distribution of Wind Angles, No Statistical Significance')
# ax.set_xlim(left=0, right=20000)

# COMMAND ----------

# MAGIC %md
# MAGIC Wind speed is next, and is what we expect. Most of the data shows low wind speeds with a long positive tail of high wind speeds. We will focus on the high wind speeds as indicative of adverse wind conditions.

# COMMAND ----------

weather_data_trimmed_full_USonly2.createOrReplaceTempView("weather_data_trimmed_full_USonly2")
tot = weather_data_trimmed_full_USonly2.count()
i = 'WND_speed'
table_equivalent = sqlContext.sql(f"SELECT {i}, count({i}), bround(count({i})/{tot} * 100, 2) AS Percentage FROM weather_data_trimmed_full_USonly2 GROUP BY {i} ORDER BY count({i}) DESC")
display(table_equivalent)

# for c in weird_columns:
fig, ax = plt.subplots()
hist(ax, weather_data_trimmed_full_USonly2.select(i), bins = 50, color=['red'])
ax.set_xlabel('Wind Speed (in Meters per Second)')
ax.set_ylabel('Frequency')
ax.set_title('Positively Skewed Wind Speed Data, 40% of readings <3.1 meters per second')
# ax.set_xlim(left=0, right=20000)

# COMMAND ----------

# MAGIC %md
# MAGIC WND_type does not have much significance; 98% of the data shows Normal wind conditions, with 2% of conditions meaning "Variable". "Variable" wind conditions may not necessarily mean good nor bad weather. "In the U.S., the criterion for a variable wind is: wind speed greater than 6 kt and direction varies by 60 degrees or more. If the wind is >1 kt but <6 kt, the wind direction may be replaced by ”VRB” followed by the speed or reported as observed." [Source] No other WND_type shows up here.
# MAGIC 
# MAGIC [Source] https://aviation.stackexchange.com/questions/38528/what-are-the-criteria-for-variable-in-metar-reports

# COMMAND ----------

# Seeing if WND_type has any significance:
weather_data_trimmed_full_USonly2.createOrReplaceTempView("weather_data_trimmed_full_USonly2")
tot = weather_data_trimmed_full_USonly2.count()
table_equivalent = sqlContext.sql(f"SELECT WND_type, count(WND_type), bround(count(WND_type)/{tot} * 100, 2) AS Percentage FROM weather_data_trimmed_full_USonly2 GROUP BY WND_type ORDER BY count(WND_type) DESC")
display(table_equivalent)

# COMMAND ----------

# MAGIC %md
# MAGIC This ceiling dimension reading is very interesting in terms of the bifrucation of the data. We interpreted this as: 58% of the time, we had the maximum visibility (good weather conditions), and we had some cases where there was low visibility. However, we expected there to be more instances of higher visibility - maybe around 90% of the time, it should be maximum visibility.

# COMMAND ----------

weather_data_trimmed_full_USonly2.createOrReplaceTempView("weather_data_trimmed_full_USonly2")
tot = weather_data_trimmed_full_USonly2.count()
i = 'CIG_ceiling_height_dimension'
table_equivalent = sqlContext.sql(f"SELECT {i}, count({i}), bround(count({i})/{tot} * 100, 2) AS Percentage FROM weather_data_trimmed_full_USonly2 GROUP BY {i} ORDER BY count({i}) DESC")
display(table_equivalent)

# weird_columns = ['VIS_distance_dimension', 'CIG_ceiling_height_dimension']
# for c in weird_columns:
fig, ax = plt.subplots()
hist(ax, weather_data_trimmed_full_USonly2.select(i), bins = 50, color=['red'])
ax.set_xlabel('Ceiling Height (in Meters)')
ax.set_ylabel('Frequency')
ax.set_title('Brifrucated Data with 58% Equal to the Max Ceiling Height Available (22K meters)')
# ax.set_xlim(left=0, right=20000)

# COMMAND ----------

# MAGIC %md
# MAGIC This "Visibility" reading also was very confusing to interpret. 83.5% of the dataset had high visibility of 16K meters (implying good weather) vs. 58% in the previous ceiling height dimension dataset. They do not line up. In addition, <0.1% of this dataset had very extreme positive values of around 100,000 meters, which doesn't really make a lot of sense given most of the data reads in the 16K meters zone. However, overall, we will focus on the low readings as indicative of adverse weather, and high readings as indicative of good weather.

# COMMAND ----------

# # Seeing if 'VIS_distance_dimension', 'CIG_ceiling_height_dimension' has any significance:
# bad_weather2 = weather_data_trimmed_full_USonly2.filter(col('Bad_Weather_Prediction') == 1)
# # bad_weather2.summary().toPandas()

weather_data_trimmed_full_USonly2.createOrReplaceTempView("weather_data_trimmed_full_USonly2")
tot = weather_data_trimmed_full_USonly2.count()
i = 'VIS_distance_dimension'
table_equivalent = sqlContext.sql(f"SELECT {i}, count({i}), bround(count({i})/{tot} * 100, 2) AS Percentage FROM weather_data_trimmed_full_USonly2 GROUP BY {i} ORDER BY count({i}) DESC")
display(table_equivalent)

# weird_columns = ['VIS_distance_dimension', 'CIG_ceiling_height_dimension']
# for c in weird_columns:
fig, ax = plt.subplots()
hist(ax, weather_data_trimmed_full_USonly2.select(i), bins = 50, color=['red'])
ax.set_xlabel('Visability (in Meters)')
ax.set_ylabel('Frequency')
ax.set_title('83% of Visibility Data clustered in ~16K meters with a Very Long Tail (<0.1% of data >20K meters)')
ax.set_xlim(left=0, right=20000)

# COMMAND ----------

# MAGIC %md
# MAGIC We see a normal distribution of air temperatures, with a slight negative skew/long negative tail. As we will detail in the feature creation section of the Bad Weather Predictor, the focus will be on the difference between dew and air temperatures (relative humidity), as opposed to the absolute low temperature readings of the air or  dew temperatures.

# COMMAND ----------

weather_data_trimmed_full_USonly2.createOrReplaceTempView("weather_data_trimmed_full_USonly2")
tot = weather_data_trimmed_full_USonly2.count()
i = 'TMP_air_temperature'
table_equivalent = sqlContext.sql(f"SELECT {i}, count({i}), bround(count({i})/{tot} * 100, 2) AS Percentage FROM weather_data_trimmed_full_USonly2 GROUP BY {i} ORDER BY count({i}) DESC")
display(table_equivalent)

fig, ax = plt.subplots()
hist(ax, weather_data_trimmed_full_USonly2.select(i), bins = 50, color=['red'])
ax.set_xlabel('Air Temperature (in Celcius)')
ax.set_ylabel('Frequency')
ax.set_title('Normally Distributed Air Temperatures, Below Freezing = Increased chance for Ice/Snow/Adverse Weather Conditions')
# ax.set_xlim(left=0, right=20000)

# COMMAND ----------

# MAGIC %md
# MAGIC Dew temperature produces a histogram similar to air temperature, except with a more extreme negative skew. Again, we will focus on the relative difference between the two (relative humidity), as opposed to absolute low dew temperature readings.

# COMMAND ----------

weather_data_trimmed_full_USonly2.createOrReplaceTempView("weather_data_trimmed_full_USonly2")
tot = weather_data_trimmed_full_USonly2.count()
i = 'DEW_dew_point_temperature'
table_equivalent = sqlContext.sql(f"SELECT {i}, count({i}), bround(count({i})/{tot} * 100, 2) AS Percentage FROM weather_data_trimmed_full_USonly2 GROUP BY {i} ORDER BY count({i}) DESC")
display(table_equivalent)

fig, ax = plt.subplots()
hist(ax, weather_data_trimmed_full_USonly2.select(i), bins = 50, color=['red'])
ax.set_xlabel('Dew Point Temperature (in Celcius)')
ax.set_ylabel('Frequency')
ax.set_title('Diff(Dew vs. Air Temp) predicts Probability of Rainfall; High Positive Correlation with Air Temperature (0.8)')
# ax.set_xlim(left=0, right=20000)

# COMMAND ----------

# MAGIC %md
# MAGIC Sea pressure produces a textbook normal distribution. A low sea pressure is supposed to be indicative of adverse weather conditions, so we will focus on the bottom 10 percentile of these sea pressure readings.

# COMMAND ----------

weather_data_trimmed_full_USonly2.createOrReplaceTempView("weather_data_trimmed_full_USonly2")
tot = weather_data_trimmed_full_USonly2.count()
i = 'SLP_sea_level_pressure'
table_equivalent = sqlContext.sql(f"SELECT {i}, count({i}), bround(count({i})/{tot} * 100, 2) AS Percentage FROM weather_data_trimmed_full_USonly2 GROUP BY {i} ORDER BY count({i}) DESC")
display(table_equivalent)

fig, ax = plt.subplots()
hist(ax, weather_data_trimmed_full_USonly2.select(i), bins = 50, color=['red'])
ax.set_xlabel('Sea Level Pressure (in Hectopascals)')
ax.set_ylabel('Frequency')
ax.set_title('Textbook Normal Distribution, Low Sea Level Pressure = Increased Chance for Adverse Weather Conditions')
# ax.set_xlim(left=0, right=20000)

# COMMAND ----------

# MAGIC %md
# MAGIC Using a correlation matrix, we found that air temperature and dew point temperature have a high positive correlation of 0.8. All other significant variables were relatively uncorrelated with each other (uncorrelated defined as <|0.3|). As we will discuss in the feature engineering section below, the Sea Level Pressure variable was engineered to have a negative correlation with bad weather (wasn't just a coincidence).

# COMMAND ----------

# Correlation matrix can only be done on subset of data, full dataset crashes the toPandas() function.
# Filters < 1/1/2016 for 1 year's worth of data. Correlations should hold for the most part between 1 yr or 10 yrs of data.
# toPandas() results in the collection of all records in the PySpark DataFrame to the driver program and should be done on a small subset of the data. running on larger dataset’s results in memory error and crashes the application.
# Source: https://stackoverflow.com/questions/54980417/some-of-my-columns-get-missing-when-i-use-df-corr-in-pandas
# Source: https://sparkbyexamples.com/pyspark/convert-pyspark-dataframe-to-pandas/
# Source: https://stackoverflow.com/questions/63964006/round-decimal-places-seaborn-heatmap-labels

correl_matrix = weather_data_trimmed_full_USonly2.filter(col('DATE') < "1451606400").drop("LATITUDE","LONGITUDE", "ELEVATION", "WND_direction_quality_code", "WND_speed_quality_code", "CIG_ceiling_quality_code", "VIS_distance_quality_code", "VIS_variability_code", "VIS_quality_variability_code", "TMP_air_temperature_quality_code", "DEW_dew_point_quality_code", "SLP_sea_level_pressure_quality_code")
correl_matrix = correl_matrix.toPandas()
# correl_matrix.corr()

fig, ax = plt.subplots()
sns.heatmap(correl_matrix.corr(), annot=True, fmt='.1f')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - Bad Weather Predictions are Evenly Distributed Across Dates (not clumped together).

# COMMAND ----------

# Seeing if bad weather predictions are clustered around a single date/seasonality.
temp = weather_data_trimmed_full_USonly2.filter((col('Bad_Weather_Prediction') == 1))
                                       #(col('DATE') > "1519862400") & (col('DATE') < "1559347200"))
display(temp)

fig, ax = plt.subplots()
hist(ax, temp.select('DATE'), bins = 50, color=['red'])
ax.set_xlabel('Bad Weather Unix Code Dates')
ax.set_ylabel('Frequency')
ax.set_title('Bad Weather Predictions are Evenly Distributed Across Dates')

# COMMAND ----------

# MAGIC %md
# MAGIC We plot boxplots of all our relevant features vs. our Bad Weather Predictor. Again, we use the 3 month dataset to avoid crashing Databricks with the full dataset. 
# MAGIC 
# MAGIC These boxplots serve as a check to see if we have any unaccounted features that could be predictive for bad weather - it looks like we have accounted for all significant features.

# COMMAND ----------

# Plot boxplots of bad weather. Again we use 3m because anything larger will crash the toPandas() function.
# data = weather_data_trimmed_full_USonly2.toPandas()
data = weather_data_trimmed_full_USonly2_3m.toPandas()
f, axes = plt.subplots(3,3, figsize=(20, 20))
f.suptitle('We have accounted for all features that may be predictive of Bad Weather', fontsize=20)
cols_of_int = ['WND_direction_angle','WND_speed','CIG_ceiling_height_dimension','VIS_distance_dimension','TMP_air_temperature','DEW_dew_point_temperature','SLP_sea_level_pressure']
count = 0

for i in range(3):
  for j in range(3):
    sns.boxplot(x='Bad_Weather_Prediction', y=cols_of_int[count], data=data, ax=axes[i,j])
    count += 1
    if count == len(cols_of_int):
      break

# COMMAND ----------

# MAGIC %md
# MAGIC #### End-to-End Pipeline from the Original df_weather Dataset
# MAGIC 
# MAGIC These following steps include EDA, data cleaning, feature transformation, feature engineering. The end result is a clean dataset that we "checkpointed" and saved into the blob.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### EDA / Weather Dataset Challenges:
# MAGIC First, we have a very large dataset of 630 million rows. This implies lots of compute and time cost embedded in analyzing/shuffling this large amount of data. Some parts of the EDA such as evaluating the number of nulls or empty rows/columns had to be done and generalized with the smaller dataset (3 months) as opposed to the full dataset because the full dataset would generate an out of memory error. 
# MAGIC  
# MAGIC Before any analysis could be done, there were many missing/erroneous features that we must remove. 161 out of 177 features had, on average, 50%+ missing data points (anywhere from 15-29 million missing rows out of 29,823,926 total in the 3 month data set), so we discarded those features. Next, since we were working within the US, we excluded any weather station/reading that was located outside of the US. There were 19 features that needed to be un-nested from 6 columns. Out of those 19 features, there were missing/erroneous rows that needed to be removed (coded as 999, 9999, or various iterations of 9’s). There were also quality codes that indicated that the value recorded was inaccurate (<5% of overall dataset); hence we discarded those rows as well. After additional cleaning (removing symbols), scaling (by a factor of 10), and casting, 42 million rows remained. We did not proceed with imputing missing/erroneous variables at this stage because we still have an abundance of data (42 million rows) for our model to train on. We determined that there were many columns (such as quality codes or wind angle) that were simply identification features as opposed to predictive features.

# COMMAND ----------

# --------------------------11/25/2021 FULL RERUN OF DF_WEATHER FULL DATASET TO FILTERED DF_WEATHER--------------------------
display(df_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC Unlike the "NAME" column whereby 0.75% of the data is blank, all the latitude and longitude data looks to be evenly distributed (no blanks). Therefore, we can leave in the blank "NAME" columns when we remove the non-US weather forcasts.

# COMMAND ----------

df_weather.createOrReplaceTempView("df_weather")
tot = df_weather.count()
missingnames = ['LATITUDE', 'LONGITUDE', 'NAME']
for i in missingnames:
  table_equivalent = sqlContext.sql(f"SELECT {i}, count({i}), bround(count({i})/{tot} * 100, 2) AS Percentage FROM df_weather GROUP BY {i} ORDER BY count({i}) DESC")
  display(table_equivalent)

# COMMAND ----------

df_weather.count()

# COMMAND ----------

df_weather_3m.count()

# COMMAND ----------

# MAGIC %md
# MAGIC We had to use the 3 month dataset because the full dataset will produce an out of memory error. 
# MAGIC However, even with the 3 month dataset, there are some columns and rows that are obviously full of blanks and NAs, per the following. Therefore, we can safely remove these features.

# COMMAND ----------

# Source: https://stackoverflow.com/questions/33900726/count-number-of-non-nan-entries-in-each-column-of-spark-dataframe-with-pyspark
weather_nas = df_weather_3m.select([count(when(col(c).isNull(), c)).alias(c) for c in df_weather_3m.columns])
display(weather_nas)

# COMMAND ----------

weather_blanks = df_weather_3m.select([count(when((col(c) == '' ), c)).alias(c) for c in df_weather_3m.columns])
# df_weather.select([count(when(col(c).contains('None') | \
#                             col(c).contains('NULL') | \
#                              | \
#                             col(c).isNull() | \
#                             isnan(c), c 
#                            )).alias(c)
#                     for c in df_weather_3m.columns])
display(weather_blanks)

# COMMAND ----------

columns_to_keep = ["STATION","DATE","LATITUDE","LONGITUDE","ELEVATION","NAME","REPORT_TYPE","CALL_SIGN","QUALITY_CONTROL","WND","CIG","VIS","TMP","DEW","SLP"]
df_weather_trimmed = df_weather.select(*columns_to_keep)
display(df_weather_trimmed)

# COMMAND ----------

# MAGIC %md
# MAGIC Unnest the data that have multiple columns' worth of data inside of 1 column

# COMMAND ----------

# Split the columns with multiple columns of data inside 1 column
weather_split = df_weather_trimmed.select("STATION","DATE","LATITUDE","LONGITUDE","ELEVATION","NAME", F.split('WND', ',').alias('WND'), F.split('CIG', ',').alias('CIG'), F.split('VIS', ',').alias('VIS'), F.split('TMP', ',').alias('TMP'), F.split('DEW', ',').alias('DEW'), F.split('SLP', ',').alias('SLP'))
display(weather_split)

# COMMAND ----------

# MAGIC %md
# MAGIC Unnest the data into their own column and rename the new columns.

# COMMAND ----------

# weather_split.printSchema()
# Source: https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PySpark_SQL_Cheat_Sheet_Python.pdf
# Source: https://stackoverflow.com/questions/49650907/split-column-of-list-into-multiple-columns-in-the-same-pyspark-dataframe
weather_split2 = weather_split.select([weather_split.STATION] + [weather_split.DATE] + [weather_split.LATITUDE] + [weather_split.LONGITUDE] + [weather_split.ELEVATION] + [weather_split.NAME] + 
                                      [weather_split.WND[i] for i in range(len(weather_split.select('WND').take(1)[0][0]))] + 
                                      [weather_split.CIG[i] for i in range(len(weather_split.select('CIG').take(1)[0][0]))] + 
                                      [weather_split.VIS[i] for i in range(len(weather_split.select('VIS').take(1)[0][0]))] + 
                                      [weather_split.TMP[i] for i in range(len(weather_split.select('TMP').take(1)[0][0]))] + 
                                      [weather_split.DEW[i] for i in range(len(weather_split.select('DEW').take(1)[0][0]))] + 
                                      [weather_split.SLP[i] for i in range(len(weather_split.select('SLP').take(1)[0][0]))])

# Source: https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
# Assuming its in order from page 8-12
weather_split2 = weather_split2.withColumnRenamed('WND[0]', 'WND_direction_angle')
weather_split2 = weather_split2.withColumnRenamed('WND[1]', 'WND_direction_quality_code')
weather_split2 = weather_split2.withColumnRenamed('WND[2]', 'WND_type')
weather_split2 = weather_split2.withColumnRenamed('WND[3]', 'WND_speed')
weather_split2 = weather_split2.withColumnRenamed('WND[4]', 'WND_speed_quality_code')
weather_split2 = weather_split2.withColumnRenamed('CIG[0]', 'CIG_ceiling_height_dimension')
weather_split2 = weather_split2.withColumnRenamed('CIG[1]', 'CIG_ceiling_quality_code')
weather_split2 = weather_split2.withColumnRenamed('CIG[2]', 'CIG_ceiling_determination_code')
weather_split2 = weather_split2.withColumnRenamed('CIG[3]', 'CIG_ceiling_and_visibility_okay_CAVOK')
weather_split2 = weather_split2.withColumnRenamed('VIS[0]', 'VIS_distance_dimension')
weather_split2 = weather_split2.withColumnRenamed('VIS[1]', 'VIS_distance_quality_code')
weather_split2 = weather_split2.withColumnRenamed('VIS[2]', 'VIS_variability_code')
weather_split2 = weather_split2.withColumnRenamed('VIS[3]', 'VIS_quality_variability_code')
weather_split2 = weather_split2.withColumnRenamed('TMP[0]', 'TMP_air_temperature')
weather_split2 = weather_split2.withColumnRenamed('TMP[1]', 'TMP_air_temperature_quality_code')
weather_split2 = weather_split2.withColumnRenamed('DEW[0]', 'DEW_dew_point_temperature')
weather_split2 = weather_split2.withColumnRenamed('DEW[1]', 'DEW_dew_point_quality_code')
weather_split2 = weather_split2.withColumnRenamed('SLP[0]', 'SLP_sea_level_pressure')
weather_split2 = weather_split2.withColumnRenamed('SLP[1]', 'SLP_sea_level_pressure_quality_code')
display(weather_split2)

# COMMAND ----------

cols_to_convert_into_float = ['WND_direction_angle','WND_direction_quality_code','WND_speed','WND_speed_quality_code','CIG_ceiling_height_dimension','CIG_ceiling_quality_code','VIS_distance_dimension','VIS_distance_quality_code','VIS_variability_code','VIS_quality_variability_code','TMP_air_temperature_quality_code','DEW_dew_point_quality_code','SLP_sea_level_pressure','SLP_sea_level_pressure_quality_code']
weather_split3 = weather_split2
# Source: https://stackoverflow.com/questions/40478018/pyspark-dataframe-convert-multiple-columns-to-float
for col_name in cols_to_convert_into_float:
    weather_split3 = weather_split3.withColumn(col_name, col(col_name).cast('float'))

# COMMAND ----------

weather_split3.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC - Include US rows only since we are only interested in US flight delays.
# MAGIC - Include "blanks" based on the NAME column since we will use the coordinates to triangulate where the weather station is located.
# MAGIC - Exclude erroneous/missing data in **relevant** features only.

# COMMAND ----------

df_weather_trimmed = weather_split3
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
# create TempView to allow SQL queries
df_weather_trimmed.createOrReplaceTempView("df_weather_trimmed")

# tot = df_weather_trimmed.count()
# for i in df_weather_trimmed.schema.names:
#   table_equivalent = sqlContext.sql(f"SELECT {i}, count({i}), bround(count({i})/{tot} * 100, 2) AS Percentage FROM df_weather_trimmed GROUP BY {i} ORDER BY count({i}) DESC")
#   display(table_equivalent)

df_weather_trimmed2 = df_weather_trimmed

# Filter out the missing rows with spark.sql, save it as weather_trimmed2
# create TempView to allow SQL queries
# Source: https://spark.apache.org/docs/2.2.0/sql-programming-guide.html
df_weather_trimmed2.createOrReplaceTempView("df_weather_trimmed2")
# Match US only: either " US", ", US", or blanks but they have latitude and longitude information (4,715,523 are blank)
# Total weather rows so far: 126,981,690 -> 128,868,028 (picked up 1m blanks)
# Comp to total airline rows we need to match against: 31,000,000
# Source: https://learnsql.com/blog/using-like-match-patterns-sql/
# Source: https://stackoverflow.com/questions/66945642/sql-match-last-two-characters-in-a-string
df_weather_trimmed2 = sqlContext.sql("SELECT * FROM df_weather_trimmed2 WHERE (right(NAME, 2) = 'US' OR NAME = '') AND WND_direction_angle != 999 AND WND_direction_quality_code in (0, 1, 4, 5, 9) AND WND_speed != 9999 AND WND_speed_quality_code in (0, 1, 4, 5, 9) AND CIG_ceiling_height_dimension != 99999 AND CIG_ceiling_quality_code in (0, 1, 4, 5, 9) AND VIS_distance_dimension != 999999 AND VIS_distance_quality_code in (0, 1, 4, 5, 9) AND TMP_air_temperature_quality_code in (0, 1, 4, 5, 9) AND DEW_dew_point_quality_code in (0, 1, 4, 5, 9) AND SLP_sea_level_pressure_quality_code in (0, 1, 4, 5, 9)")
display(df_weather_trimmed2)

# COMMAND ----------

# MAGIC %md
# MAGIC - 128,868,028 rows remain (US only, non-erroneous, non-missing rows in relevant features)

# COMMAND ----------

# Verify ALL US ONLY or Blank.
tot = df_weather_trimmed2.count()
print(tot)
missingnames = ['NAME']
df_weather_trimmed2.createOrReplaceTempView("df_weather_trimmed2")
for i in missingnames:
  table_equivalent = sqlContext.sql(f"SELECT {i}, count({i}), bround(count({i})/{tot} * 100, 2) AS Percentage FROM df_weather_trimmed2 GROUP BY {i} ORDER BY count({i}) DESC")
  display(table_equivalent)

# COMMAND ----------

# MAGIC %md
# MAGIC - We added the unix timestamp instead of regular timestamp so that this dataset could be joined with the airlines dataset.
# MAGIC 
# MAGIC - We removed the + symbol in the temperature columns, so that they could be casted as floats and removed missing values.
# MAGIC 
# MAGIC - We then scaled the temperatures by 10 per the documentation. (https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf)

# COMMAND ----------

# Add unix timestamp
df_weather_trimmed2 = df_weather_trimmed2.withColumn('DATE',F.unix_timestamp(F.col('DATE')))
# Remove the "+" symbol in 2 columns
df_weather_trimmed2 = df_weather_trimmed2.withColumn('TMP_air_temperature', regexp_replace('TMP_air_temperature', '\\+', ''))
df_weather_trimmed2 = df_weather_trimmed2.withColumn('DEW_dew_point_temperature', regexp_replace('DEW_dew_point_temperature', '\\+', ''))
# Cast those 2 columns as floats.
df_weather_trimmed2 = df_weather_trimmed2.withColumn('TMP_air_temperature', col('TMP_air_temperature').cast('float'))
df_weather_trimmed2 = df_weather_trimmed2.withColumn('DEW_dew_point_temperature', col('DEW_dew_point_temperature').cast('float'))
# Remove missing values
df_weather_trimmed2.createOrReplaceTempView("df_weather_trimmed2")
df_weather_trimmed2 = sqlContext.sql("SELECT * FROM df_weather_trimmed2 WHERE TMP_air_temperature != 9999 AND DEW_dew_point_temperature != 9999 AND SLP_sea_level_pressure != 99999")
# Scale temperatures by 10
df_weather_trimmed2 = df_weather_trimmed2.withColumn('TMP_air_temperature', col('TMP_air_temperature')/10)
df_weather_trimmed2 = df_weather_trimmed2.withColumn('DEW_dew_point_temperature', col('DEW_dew_point_temperature')/10)
df_weather_trimmed2 = df_weather_trimmed2.withColumn('WND_speed', col('WND_speed')/10)
df_weather_trimmed2 = df_weather_trimmed2.withColumn('SLP_sea_level_pressure', col('SLP_sea_level_pressure')/10)
df_weather_trimmed2.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC - 42,089,829 rows remaining after removing missing Temp, Dew, and Sea level pressures.

# COMMAND ----------

display(df_weather_trimmed2)
df_weather_trimmed2.count()

# COMMAND ----------

# Save to blob
# This takes 8 minutes to run.
df_weather_trimmed2.write.mode("overwrite").parquet(f"{blob_url}/weather_data_trimmed_full_USonly")

# COMMAND ----------

# MAGIC %md
# MAGIC We save our first checkpoint as the cleaned dataset with 42 million rows.
# MAGIC 
# MAGIC Given the data we have, we will now add a binary feature that "predicts" if weather is bad or not (1=bad weather predicted, 0=good weather pred)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature Engineering
# MAGIC We created a binary feature called “Bad_Weather_Prediction.” After discarding the missing/erroneous rows in the usable feature set, we were left with a total of 42 million rows. First, we established an upper limit. The max number of days that it rains per year in the US is 45% of all days [1]. Therefore, our predictor should not exceed 18.9 million bad weather data point predictions (42*0.45), since that would assume the max amount of rainfall in every state.
# MAGIC  
# MAGIC We focused on the air and dew temperatures first. When air temperature and dew point temperatures are very close, the air has a high relative humidity and a lower chance of rain. When there is a large difference between air and dew point temperatures, this indicates lower relative humidity and a higher chance of rain [2]. These sources state that a reading above 50% elevates the chance of rain [3, 4]. We thought this was a basic filter that predicted whether bad weather was coming, and more filters can be added on top of this to make it more robust.
# MAGIC  
# MAGIC Next, we developed three additional criteria in which one of three would need to be met in order to firmly predict that bad weather is indeed on its way. Wind speeds in excess of 34 mph (15.2 meters per second) is indicative of bad weather [5]. Low sea pressure also indicates bad weather, so we defined “low sea pressure” as any reading in the bottom 10% of the dataset [6]. The required visibility required for a plane to take off is 1 mile or 1609.34 meters [7]. Therefore, any reading below 1609.34 meters is considered a bad weather indicator as well.
# MAGIC  
# MAGIC Additional considerations include considering below freezing (0 degrees Celsius) as bad weather, since below freezing indicates a higher chance of ice and snow on the runways. However, we felt that this was too broad of an interpretation and showed many false positives in colder areas such as Alaska and states in the Northeast. We also looked into the minimum ceiling height dimension required: “the means of egress shall have a ceiling height of not less than 7 feet 6 inches (2286 mm or 2.286 meters)” [8]. However, the rows in our dataset show a very bifurcated reading with the 50th percentile at 22,000 mm and the other half clumped around <4000 mm. We concluded that using this feature would not be as informative or predictive as the other criteria if half of the dataset (21 million) would be labeled as “bad weather.” This would also violate our upper bound assumption of 18.9 million rows.
# MAGIC  
# MAGIC With the four criteria mentioned above, the Bad_Weather_Prediction column was added: 1 for bad weather, 0 for good weather. We netted a total of 3,956,117 rows (~9.4% of the total dataset). Though not verified with rigorous scientific precision, this makes logical sense to us since we can reasonably conclude that bad weather in the US, on average, occurs around 10-15% of the time, taking into account the difference between wet and dry states, increase in global warming, etc.
# MAGIC  
# MAGIC #### Citations
# MAGIC [1] https://www.statista.com/statistics/226747/us-cities-with-the-most-rainy-days/
# MAGIC  
# MAGIC [2] https://bmcnoldy.rsmas.miami.edu/Humidity.html
# MAGIC  
# MAGIC [3] http://tornado.sfsu.edu/geosciences/classes/m356/Dewpoint.htm
# MAGIC  
# MAGIC [4] https://www.quora.com/Why-is-it-that-when-the-relative-humidity-is-more-there-will-be-a-maximum-probability-of-rain
# MAGIC  
# MAGIC [5] https://www.skyscanner.com/tips-and-inspiration/what-windspeed-delays-flights
# MAGIC  
# MAGIC [6] https://www.livescience.com/39315-atmospheric-pressure.html#:~:text=%E2%80%9CSunny%2C%E2%80%9D%20for%20instance%2C,on%20occasion%20below%2029%20inches.
# MAGIC  
# MAGIC [7] https://www.flyingmag.com/training-instrument-flight-rules-what-are-your-ifr-takeoff-minimums/#:~:text=Reasonable%20IFR%20Takeoff%20Minimums%3F,at%20what%20they%20are%20doing.
# MAGIC  
# MAGIC [8] https://codes.iccsafe.org/content/IBC2015/chapter-10-means-of-egress#:~:text=The%20means%20of%20egress%20shall,1.

# COMMAND ----------

# Create bad weather column: Print interquartile statistics to see what is considered extreme weather readings
# Source: https://mungingdata.com/apache-spark/dataframe-summary-describe/
#df_weather_trimmed_full.select("TMP_air_temperature").summary("count", "25%", "50%", "66%").show()
numbered_columns = ['WND_direction_angle','WND_direction_quality_code','WND_speed','WND_speed_quality_code','CIG_ceiling_height_dimension','CIG_ceiling_quality_code','VIS_distance_dimension','VIS_distance_quality_code','VIS_variability_code','VIS_quality_variability_code','TMP_air_temperature_quality_code','DEW_dew_point_quality_code','SLP_sea_level_pressure','SLP_sea_level_pressure_quality_code', 'TMP_air_temperature', 'DEW_dew_point_temperature']
df_weather_trimmed2.select([c for c in numbered_columns]).summary("count", "mean", "stddev", "min", "10%", "25%", "50%", "75%", "90%", "max").toPandas()

# COMMAND ----------

# Create Bad Weather column
# Max rain = 45% of days per year, so we should not exceed 42*0.45 = 18.9M predicted "bad weather" points
# Source: https://www.statista.com/statistics/226747/us-cities-with-the-most-rainy-days/
# 20,422,331 rows out of 42,089,829 total of bad weather, | & >0.75 for relative humidity
# 3,956,117 rows out of 42,089,829 total of bad weather, only relative humidity is a &, all others are |'s, >0.5 for relative humidity
# 10,527 rows out of 42,089,829 total of bad weather, & & >0.5 for relative humidity

# High wind speeds; High wind speed definition: 34+ mph is bad
  # Source: https://www.skyscanner.com/tips-and-inspiration/what-windspeed-delays-flights 
# Low sea pressure = Bad
  # Source: https://www.livescience.com/39315-atmospheric-pressure.html#:~:text=%E2%80%9CSunny%2C%E2%80%9D%20for%20instance%2C,on%20occasion%20below%2029%20inches.
# Visibility: Sounds like there is no requirement, as long as visibility >1 mile, you are cleared to take off. 
  # Source: https://www.flyingmag.com/training-instrument-flight-rules-what-are-your-ifr-takeoff-minimums/#:~:text=Reasonable%20IFR%20Takeoff%20Minimums%3F,at%20what%20they%20are%20doing.
# Small Diff between dew and air temp = Higher chance of rain: 
  # Source http://tornado.sfsu.edu/geosciences/classes/m356/Dewpoint.htm: 
  # When air temperature and dew point temperatures are very close, the air has a high relative humidity. The opposite is true when there is a large difference between air and dew point temperatures, which indicates air with lower relaitve humidity.
  # Relative Humidity of 20% is fairly dry. Relative humidity of 50% or higher can cause rain. Relative Humidity of 100%, it is raining, usually. 
  # Source: https://www.quora.com/Why-is-it-that-when-the-relative-humidity-is-more-there-will-be-a-maximum-probability-of-rain
  # Formula Source: https://bmcnoldy.rsmas.miami.edu/Humidity.html
  # =(EXP((17.625*-5)/(243.04+-5))/EXP((17.625*-1.6)/(243.04+-1.6)))
# TBD: Below freezing (high chance of snow/ice) for temperatures
# TBD: Minimum ceiling height dimension required: The means of egress shall have a ceiling height of not less than 7 feet 6 inches (2286 mm or 2.286 meters).
  # Source: https://codes.iccsafe.org/content/IBC2015/chapter-10-means-of-egress#:~:text=The%20means%20of%20egress%20shall,1.

bad_weather = df_weather_trimmed2.filter(((exp((17.625*col('DEW_dew_point_temperature'))/(243.04+col('DEW_dew_point_temperature')))/exp((17.625*col('TMP_air_temperature'))/(243.04+col('TMP_air_temperature')))) > 0.5) & ((col('WND_speed') >= 15.2) | (col('SLP_sea_level_pressure') <= 1006.6) | (col('VIS_distance_dimension') <= 1609.344)))
bad_weather.summary().toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC - Add the new feature into the dataframe and save to blob.

# COMMAND ----------

# Add the column into dataframe
# Confirm 42,089,829 rows
df_weather_trimmed3 = df_weather_trimmed2.withColumn('Bad_Weather_Prediction', F.when(((exp((17.625*col('DEW_dew_point_temperature'))/(243.04+col('DEW_dew_point_temperature')))/exp((17.625*col('TMP_air_temperature'))/(243.04+col('TMP_air_temperature')))) > 0.5) & ((col('WND_speed') >= 15.2) | (col('SLP_sea_level_pressure') <= 1006.6) | (col('VIS_distance_dimension') <= 1609.344)), 1).otherwise(0))
display(df_weather_trimmed3)
df_weather_trimmed3.count()

# COMMAND ----------

# Write to blob
df_weather_trimmed3.write.mode("overwrite").parquet(f"{blob_url}/weather_data_trimmed_full_USonly2")
