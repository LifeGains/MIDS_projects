# Databricks notebook source
# MAGIC %md
# MAGIC # EDA on Reporting Carrier On-Time Performance

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook focuses on the EDA for the Reporting Carrier On-Time Performance dataset from the Bureau of Transportation Statistics (BTS).
# MAGIC 
# MAGIC For the first phase of the project we will focus on flights departing from two major US airports (ORD (Chicago O’Hare) and ATL (Atlanta) in the first half of 2015 (six months of data). For the final phase of the project we will focus on the entire flight data departing from all major US airports for the 2015-2019 timeframe.

# COMMAND ----------

# imports
import re
import ast
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.functions import col

# COMMAND ----------

# init script to create the blob URL
blob_container = "team07"
storage_account = "team07"
secret_scope = "team07"
secret_key = "team07"
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"

# generates the SAS token
spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

# load 2015 Q1 data
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")
# create TempView to allow SQL queries
df_airlines.createTempView("airlines")

# COMMAND ----------

# MAGIC %md
# MAGIC ## More than 20% of flights delay more than 15 minutes

# COMMAND ----------

# query departure delay data
delay_hist_df = spark.sql("SELECT dep_delay FROM airlines ORDER BY dep_delay ASC").na.drop().toPandas()
# set-up a new figure with white facecolor
plt.figure(figsize=(6, 4), facecolor='white')
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
plt.text(x=2, y=0.99, s='23% of flights delayed more than 15 min', fontsize=15, color='#3D6197', fontweight='bold')
# set y axis params
plt.ylabel('Frequency', fontsize=12, labelpad=10, color='#75686D')
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
# MAGIC ## Majority of delays caused by own carrier or late aircrafts

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
plt.title(r'Root causes for departure delays, 1Q2015', pad=20, fontsize=15, x=0.42, color='#75686D')
# place vertical and horizontal lines
plt.text(x=-0.4, y=750, s='Majority of delays caused by carrier or late aircrafts', fontsize=15, color='#3D6197', fontweight='bold')
# set y axis params
plt.ylabel('Total delay in 1,000 min', fontsize=12, labelpad=10, color='#75686D')
plt.yticks(np.arange(0.0, 1000, step=200))
plt.ylim((0.0, 800))
# display results
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delays are not uniformly distributed over the days

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
plt.figure(figsize=(6, 4), facecolor='white')
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
plt.text(x=5, y=0.6, s='Delays tend to concentrate on specific days', fontsize=15, color='#3D6197', fontweight='bold')
# set y axis params
plt.ylabel('Pct of flights with delays >15min', fontsize=12, labelpad=9, color='#75686D')
plt.yticks(np.arange(0.0, 0.7, step=0.1))
plt.ylim((0.0, 0.6))
# x axis params
every_nth = 10
for n, label in enumerate(ax.xaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)
ax.tick_params(axis='both', which='both', length=0)
# display results
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delays are not uniformly distributed among the airlines

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
plt.figure(figsize=(8, 4), facecolor='white')
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
plt.title(r'Delay incidence per airline in 1Q2015', pad=22, fontsize=15, x=0.27, color='#75686D')
# place vertical and horizontal lines
plt.text(x=-0.4, y=0.48, s='Delays tend to concentrate on specific airlines', fontsize=15, color='#3D6197', fontweight='bold')
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
# MAGIC ## Delays are not uniformly distributed among the origin airports

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
for x in pct_delayed_by_airport_df.index:
  ax.bar(str(x), pct_delayed_by_airport_df.loc[x, 'pct_delayed'], color='#3D6197')
# place plot title
plt.title(r'Delay incidence per origin airport in 1Q2015', pad=22, fontsize=15, x=0.43, color='#75686D')
# place vertical and horizontal lines
plt.text(x=-0.4, y=0.48, s='Delays tend to concentrate on specific origin airports', fontsize=15, color='#3D6197', fontweight='bold')
# set y axis params
plt.ylabel('Pct of flights with delays >15min', fontsize=12, labelpad=9, color='#75686D')
plt.yticks(np.arange(0.0, 0.6, step=0.1))
plt.ylim((0.0, 0.5))
# set x axis params
plt.xlabel('Origin airport', fontsize=12, labelpad=10, color='#75686D')
# display results
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delays are not uniformly distributed among the destination airports

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
plt.figure(figsize=(8, 4), facecolor='white')
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
plt.title(r'Delay incidence per destination airport in 1Q2015', pad=22, fontsize=15, x=0.37, color='#75686D')
# place vertical and horizontal lines
plt.text(x=5, y=0.49, s='Delays tend to concentrate on specific destination airports', fontsize=15, color='#3D6197', fontweight='bold')
# set y axis params
plt.ylabel('Pct of flights with delays >15min', fontsize=12, labelpad=9, color='#75686D')
plt.yticks(np.arange(0.0, 0.6, step=0.1))
plt.ylim((0.0, 0.5))
# set x axis params
plt.xlabel('Destination airport', fontsize=12, labelpad=10, color='#75686D')
plt.xticks([])
# display results
plt.show()

# COMMAND ----------


