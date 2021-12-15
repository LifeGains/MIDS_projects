# Databricks notebook source
# Load 2015 Q1 for Flights & Weather
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

# Inspect the Mount's Final Project folder 
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project"))

# COMMAND ----------

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

# COMMAND ----------

# Load 2015 Q1 for Flights & Weather
blob_container = "team07" # The name of your container created in https://portal.azure.com
storage_account = "team07" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team07" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team07" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"
#df_airlines_6m = spark.read.parquet(f"{blob_url}/df_airlines_final_6m_v1")
#df_weather_6m = spark.read.parquet(f"{blob_url}/weather_data_trimmed_full_USonly2").filter(col('DATE') < "1435733999")

df_airlines_all = spark.read.parquet(f"{blob_url}/df_airlines_final_full_v7")
df_weather_all = spark.read.parquet(f"{blob_url}/weather_data_trimmed_full_USonly2")
display(df_airlines_all)
display(df_weather_all)

# COMMAND ----------

#df_weather_all.agg({"DATE": "max"}).collect()[0][0]

# COMMAND ----------

#df_airlines_all.agg({"FL_DATE": "max"}).collect()[0][0]

# COMMAND ----------

df_airlines_all.printSchema()

# COMMAND ----------

df_weather_all.printSchema()

# COMMAND ----------

from pyspark.sql.window import Window

# COMMAND ----------

w = Window.orderBy(sf.lit('A'))
df_airlines_all = df_airlines_all.withColumn('row_num',sf.row_number().over(w))

# COMMAND ----------

from pyspark.sql.types import StringType
df_weather_all = df_weather_all.withColumn("lat_lng", concat(df_weather_all.LATITUDE.cast(StringType()), sf.lit(" ") , df_weather_all.LONGITUDE.cast(StringType())))

# COMMAND ----------

uniq_weather_station_lat_lng_list = df_weather_all.select("lat_lng", "STATION").distinct().collect()

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

# COMMAND ----------

uniq_cities = [city.ORIGIN_CITY_NAME for city in uniq_cities_rows]

# COMMAND ----------

city_to_station_map = {city: get_matching_weather_station(city) for city in uniq_cities}
city_to_station_map

# COMMAND ----------

lookup = udf(lambda x: city_to_station_map[x], returnType= StringType())

# COMMAND ----------

df_airlines_all = df_airlines_all.withColumn("weather_station", lookup(df_airlines_all.ORIGIN_CITY_NAME))

# COMMAND ----------

condition = [df_airlines_all.weather_station == df_weather_all.STATION]

# COMMAND ----------

df_joined = df_airlines_all.join(df_weather_all,on = condition, how="cross")

# COMMAND ----------

df_joined.printSchema()

# COMMAND ----------

df_joined = df_joined.withColumn('time_diff', df_joined.depart_unix_timestamp - df_joined.DATE)

# COMMAND ----------

df_joined = df_joined.filter((df_joined.time_diff >= 7200) & (df_joined.time_diff <= 14400))

# COMMAND ----------

df_joined_subset = df_joined.groupBy('row_num').min('time_diff')

# COMMAND ----------

df_joined_subset = df_joined_subset.withColumnRenamed("row_num","row_number")
df_joined_subset = df_joined_subset.withColumnRenamed("min(time_diff)","time_difference")

# COMMAND ----------

df_joined_subset.count()

# COMMAND ----------

condition_for_final_join = [(df_joined_subset.row_number == df_joined.row_num) & (df_joined_subset.time_difference == df_joined.time_diff)]

# COMMAND ----------

df_joined_latest_weather = df_joined_subset.join(df_joined, on = condition_for_final_join, how = "inner")

# COMMAND ----------

df_joined_latest_weather.count()

# COMMAND ----------

#df_joined_latest_weather.write.mode("overwrite").parquet(f"{blob_url}/joined_data_all_v1")
