# Databricks notebook source
# MAGIC %md
# MAGIC ## Joined Data
# MAGIC This notebook the workflow of the joined dataframe. We have an airlines data table `df_airlines` with ~31 million rows. The weather data table has about ~41 million rows. 
# MAGIC 
# MAGIC ### Challenge: 
# MAGIC 
# MAGIC The biggest challenge was to find a common column between the two data tables to join. The airlines data table has a weather station column, but the name of the weather station doesnot always cooresspond to the weather stations in the weather table. In this notebook, we explain each step to solve this problem.

# COMMAND ----------

blob_container = "team07" # The name of your container created in https://portal.azure.com
storage_account = "team07" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team07" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team07" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

# Generates the SAS token
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

# Inspect the Mount's Final Project folder 
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project"))

# COMMAND ----------

#Import necessary libraries 
import pyspark
from pyspark.sql import functions as sf
from datetime import datetime 
from pyspark.sql.functions import split
import pytz
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat,col
from pyspark.sql.types import TimestampType
from pyspark.sql.types import StringType
import pandas as pd
import numpy as np
from pyspark.sql.window import Window
from pyspark.sql.types import StringType

# COMMAND ----------

# Load Data
blob_container = "team07" # The name of your container created in https://portal.azure.com
storage_account = "team07" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team07" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team07" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

df_airlines_all = spark.read.parquet(f"{blob_url}/df_airlines_final_full_v7")
df_weather_all = spark.read.parquet(f"{blob_url}/weather_data_trimmed_full_USonly2")

display(df_airlines_all)
display(df_weather_all)

# COMMAND ----------

df_airlines_all.printSchema()

# COMMAND ----------

df_weather_all.printSchema()

# COMMAND ----------

# MAGIC %md 
# MAGIC The airlines column has no identifying feature that can tag each flight as unique. So we introduce row numbers that now sets each flight as a distinct one. 

# COMMAND ----------

w = Window.orderBy(sf.lit('A'))
df_airlines_all = df_airlines_all.withColumn('row_num',sf.row_number().over(w))

# COMMAND ----------

#df_airlines_all.display()

# COMMAND ----------

# MAGIC %md
# MAGIC We want the latitude and longitude column of the weather data table together. So we concatenate them and put it in a new column named `lat_lng`

# COMMAND ----------

df_weather_all = df_weather_all.withColumn("lat_lng", concat(df_weather_all.LATITUDE.cast(StringType()), sf.lit(" ") , df_weather_all.LONGITUDE.cast(StringType())))
#display(df_weather_all)

# COMMAND ----------

# MAGIC %md
# MAGIC Taking a look at the number of unique latitude-longitude combinations we have..

# COMMAND ----------

uniq_weather_station_lat_lng_list = df_weather_all.select("lat_lng", "STATION").distinct().collect()

# COMMAND ----------

# MAGIC %md
# MAGIC We now write a method that computes the distance between two latitude-longitude pairs using the `Haversine formula`.

# COMMAND ----------

import math

def deg2rad(deg):
  return deg * (math.pi/180)

def get_distance_between_2_coords(lat1, lon1, lat2, lon2):
  # dist using Haversine formula - http://en.wikipedia.org/wiki/Haversine_formula
  R = 6371 # Radius of the earth in km
  dLat = deg2rad(lat2-lat1)
  dLon = deg2rad(lon2-lon1); 
  a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
  d = R * c # Distance in km
  return d

# COMMAND ----------

# MAGIC %md
# MAGIC In the following cell, we use Google's geocode api to find the latitude and longitude of each departing airport. We map back these coordinates to a matching weather station latitude and longitude using Haversine formula. The Havesine formula is used to determine the great-circle distance between two points on a sphere given their longitudes and latitudes. Now each departing city in the weather data has a weather station from where we can extract the hourly weather data.

# COMMAND ----------

import requests

def get_matching_weather_station(dept_city):
  possible_locations = [
    loc for loc in [
        dept_city,
        dept_city.split("/")[0] + ", " + dept_city.split(",")[1] if "/" in dept_city else None,
        dept_city.split("/")[1] + ", " + dept_city.split(",")[1] if "/" in dept_city else None,
        dept_city.split(",")[0] if "," in dept_city else None,
      ] if loc is not None
  ]
  possible_locations_generator = iter(possible_locations)
  
  def _get_lat_lng(location):
    if not location:
      return None
    try:
      result = requests.get("https://maps.googleapis.com/maps/api/geocode/json?address={}&key=AIzaSyBhXMu8-J0MTRcZj5XSPeKi-tObhAzZP9w".format(location)).json()["results"]
      return result[0]["geometry"]["location"]
    except IndexError:
      # Google can't decode this address, we'll retry!
      return _get_lat_lng(next(possible_locations_generator, None))
  
  city_coords = _get_lat_lng(next(possible_locations_generator))
  
  min_dist = math.inf
  closest_station = None
  for row in uniq_weather_station_lat_lng_list:
    weather_lat, weather_lng = row["lat_lng"].split()
    dist = get_distance_between_2_coords(float(city_coords["lat"]), float(city_coords["lng"]), float(weather_lat), float(weather_lng))
    if dist < min_dist:
      min_dist = dist
      closest_station = row["STATION"]
  return closest_station

# COMMAND ----------

uniq_cities_rows = df_airlines_all.select("ORIGIN_CITY_NAME").distinct().collect()
len(uniq_cities_rows)

# COMMAND ----------

uniq_cities = [city.ORIGIN_CITY_NAME for city in uniq_cities_rows]
uniq_cities

# COMMAND ----------

# MAGIC %md
# MAGIC Checking the Departing city to weather station number map. 

# COMMAND ----------

city_to_station_map = {city: get_matching_weather_station(city) for city in uniq_cities}
city_to_station_map

# COMMAND ----------

lookup = udf(lambda x: city_to_station_map[x], returnType= StringType())

# COMMAND ----------

# MAGIC %md
# MAGIC Adding the weather station number to the airlines data table. This column will be used to perform the join in the next steps. 

# COMMAND ----------

df_airlines_all = df_airlines_all.withColumn("weather_station", lookup(df_airlines_all.ORIGIN_CITY_NAME))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Joining Datasets
# MAGIC 
# MAGIC The first join condition only looks for the matching weather station from each dataframe. This is a cross join which returns the cartesian product of the two dataframes. The cross join is most pertinent in our case because we have multiple timestamps of weather data for a particular weather station and we wanted to take all of that in our joined dataframe. 
# MAGIC 
# MAGIC In the next part, we filter only those rows that have weather timestamps greater than 2 hours but less than 4 hours. Thus we only look at the most recent weather for each flight departure. 
# MAGIC 
# MAGIC The final join is an inner join which takes only the filtered rows for latest weather and joins it back with the original dataset. We have ~26 million rows in the final joined dataframe. 

# COMMAND ----------

condition = [df_airlines_all.weather_station == df_weather_all.STATION]

# COMMAND ----------

df_joined = df_airlines_all.join(df_weather_all,on = condition, how="cross")
# display(df_joined)

# COMMAND ----------

df_joined.printSchema()

# COMMAND ----------

df_joined = df_joined.withColumn('time_diff', df_joined.depart_unix_timestamp - df_joined.DATE)

# COMMAND ----------

#df_joined.display()

# COMMAND ----------

df_joined = df_joined.filter((df_joined.time_diff >= 7200) & (df_joined.time_diff <= 14400))

# COMMAND ----------

df_joined_subset = df_joined.groupBy('row_num').min('time_diff')

# COMMAND ----------

df_joined_subset = df_joined_subset.withColumnRenamed("row_num","row_number")
df_joined_subset = df_joined_subset.withColumnRenamed("min(time_diff)","time_difference")
#df_joined_subset.display()

# COMMAND ----------

#df_joined_subset.count()

# COMMAND ----------

condition_for_final_join = [(df_joined_subset.row_number == df_joined.row_num) & (df_joined_subset.time_difference == df_joined.time_diff)]

# COMMAND ----------

df_joined_latest_weather = df_joined_subset.join(df_joined, on = condition_for_final_join, how = "inner")

# COMMAND ----------

#df_joined_latest_weather.count()

# COMMAND ----------

#df_joined_latest_weather.display()

# COMMAND ----------

# MAGIC %md
# MAGIC At this point we wanted to create a checkpoint and save the data. We save it on the blob as a parquet file. 

# COMMAND ----------

#df_joined_latest_weather.write.mode("overwrite").parquet(f"{blob_url}/joined_data_all_v1")

# COMMAND ----------

# Load Data
blob_container = "team07" # The name of your container created in https://portal.azure.com
storage_account = "team07" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team07" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team07" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

df_joined_reload = spark.read.parquet(f"{blob_url}/joined_data_all_v1")

# COMMAND ----------

# MAGIC %md
# MAGIC We faced a major problem here. The joined dataframe gets saved as a parquet file without any complaints but when we try to use the checkpoint data and reload it `here` or on `another notebook` for further analysis, the count of the data frame is drastically lower. We manage to save only ~2 million rows to the joined dataframe. For the rest of the notebook, we will use that dataframe as our complete dataset. 

# COMMAND ----------

df_joined_reload.count()

# COMMAND ----------

# Load Data
blob_container = "team07" # The name of your container created in https://portal.azure.com
storage_account = "team07" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team07" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team07" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

df_joined_v1 = spark.read.parquet(f"{blob_url}/joined_data_all_v1")

# COMMAND ----------

df_joined_v1.count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### EDA on df_joined_latest_weather

# COMMAND ----------

df_joined_v1_pd = df_joined_v1.toPandas()

# COMMAND ----------

df_joined_v1_pd.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ### Missing Data

# COMMAND ----------

import matplotlib.pyplot as plt
missing_cols = df_joined_v1_pd.isnull().sum()
plot_cols = missing_cols[missing_cols>1000] # drop 0 count cols
plot_cols.sort_values(inplace=True)
plot_cols.plot.bar(figsize=(12,8))
plt.xlabel("Feature",fontsize=14)
plt.ylabel("Missing values",fontsize=14)
plt.title("Barchart of counts of missing values",fontsize=16)
plt.show()

# COMMAND ----------

df_temp =  pd.DataFrame(df_joined_v1_pd.groupby('ORIGIN_CITY_NAME').count()['Bad_Weather_Prediction'])

# COMMAND ----------

# MAGIC %md
# MAGIC Airports that have more than 10000 bad weather predictions. 

# COMMAND ----------

df_temp[df_temp['Bad_Weather_Prediction'] > 10000].sort_values(by=['Bad_Weather_Prediction'],ascending=False)

# COMMAND ----------

df_joined_v1_pd.describe()

# COMMAND ----------

len(df_joined_v1_pd['TAIL_NUM'].unique())

# COMMAND ----------

df_temp1 =  pd.DataFrame(df_joined_v1_pd.groupby('TAIL_NUM').count()['Bad_Weather_Prediction'])
df_temp1
