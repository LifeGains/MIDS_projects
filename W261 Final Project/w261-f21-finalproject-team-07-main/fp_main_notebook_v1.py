# Databricks notebook source
# MAGIC %md
# MAGIC # Prediction of Flight Delays in the United States using Machine Learning Algorithms at Scale
# MAGIC `MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Summer 2021`
# MAGIC 
# MAGIC **`Team 07: Atreyi Dasmahapatra, Lucas Barbosa, Kanika Mahajan, Kevin Fu`**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Abstract
# MAGIC 
# MAGIC Flight delays have costed the American economy an estimated $28 billion in 2018, including direct and indirect costs to airlines and passengers [3]. 
# MAGIC In this project we use machine learning algorithms to predict if a flight will have a departure delay from its origin airport using the data from the Office of Airline Information, Bureau of Transportation Statistics (BTS). It is a common assumption that flights are most likely to be delayed due to unfavorable weather conditions. We test this hypothesis using weather data from the the Integrated Surface Data (ISD). 
# MAGIC 
# MAGIC Our computations show that the most likely cause of a flight delay is due to the delays of the incoming flight. The rest of the report is designed as follows: In the Introduction part, we explain the problem statement and describe historical work that has aimed to explain flight delays. The next part includes a discussion of the datasets in depth and explain the exploratory data analysis. For any predictive machine learning algorithm, feature engineering is of utmost importance. We devote our next section to the feature engineering we implemented. We then move on to explain our chosen machine learning algorithms. The final two sections summarise our main results, challenges we faced during the project and our conclusions.   

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table of Contents
# MAGIC 
# MAGIC 1. Introduction
# MAGIC 
# MAGIC 2. Description of datasets and exploratory data analysis
# MAGIC 
# MAGIC 3. Feature Engineering / Feature Creation
# MAGIC 
# MAGIC 4. Algorithm Exploration & Implementation
# MAGIC 
# MAGIC 5. Results and Discussion
# MAGIC 
# MAGIC 6. Challenges faced and future work
# MAGIC 
# MAGIC 7. Conclusions 
# MAGIC 
# MAGIC 8. Application of course concepts

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 1. Introduction

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### - Why does flight delay prediction matter?
# MAGIC 
# MAGIC Flight delays are inevitable but still a curse of modern life, causing billions of dollars of extra costs for airlines and passengers every year. In 2017, only 76% of the flights in US arrived on time [1]. Between 2013 and 2019, the number of flights in Europe increased by 16% but the average departure delay of European flights increased by 41% [2]. On-time flight performance is considered an important measure of the service quality of airports
# MAGIC and airlines. Delays impact airlines and passengers alike. Passengers incur extra time to get to their destinations, increasing their trip costs [4], not to mention the inconvenience of stays at airports and missed connections. Likewise, the airlines companies incur penalties, fines and additional operation costs, such as crew and aircrafts retentions in airports [4]. Delays also jeopardize airlines marketing strategies [4], since carriers rely on customers’ loyalty to support their growth and financial standing and the consumer’s choice is significantly affected by unreliable performance. Thus, it is worth trying to predict flight delays significantly before the flight is about to depart that will enable both airlines and passengers to come up with another plan of action. 
# MAGIC 
# MAGIC ### - What causes a flight to delay?
# MAGIC 
# MAGIC Since 2003, all airlines operating in the United States need to report the causes of delay within some broad categories that were created by the Air Carrier On-Time Reporting Advisory Committee [7]. The categories are:
# MAGIC 
# MAGIC * **Air Carrier:** The cause of the cancellation or delay was due to circumstances within the airline's control (e.g. maintenance or crew problems, aircraft cleaning, baggage loading, fueling, etc.).
# MAGIC * **Late-arriving aircraft:** A previous flight with same aircraft arrived late, causing the present flight to depart late.
# MAGIC * **National Aviation System (NAS):** Delays and cancellations attributable to the national aviation system that refer to a broad set of conditions, such as non-extreme weather conditions, airport operations, heavy traffic volume, and air traffic control.
# MAGIC * **Extreme Weather:** Significant meteorological conditions (actual or forecasted) that, in the judgment of the carrier, delays or prevents the operation of a flight such as tornado, blizzard or hurricane.
# MAGIC * **Security:** Delays or cancellations caused by evacuation of a terminal or concourse, re-boarding of aircraft because of security breach, inoperative screening equipment and/or long lines in excess of 29 minutes at screening areas.
# MAGIC 
# MAGIC Data reported by the Bureau of Transportation Statistics shows that in 2020 the most commom causes for delay in US were Air Carrier delay (41%), Aircraft arriving late (30.2%), NAS Delay (21.6%), Extreme Weather (6.9%) and Security (0.2%) [7].
# MAGIC 
# MAGIC ### - Why is it so difficult to predict flight delays?
# MAGIC 
# MAGIC Commercial aviation is a complex transportation system. It deals with a sophisticated and interconnected origin-destination network that needs orchestration to provide smooth operations [4]. Every day, airlines do plan various schedules for aircrafts, pilots and flight attendants. Delays have ripple effects, where delays in one part of the network quickly propagate to the others. Turn times between flight legs are constantly minimized by airlines in order to fully utilize expensive resources, so there is not much slack in the schedule to absorb delays [5]. Crews are also commonly shared between flight legs [6] and have restrictions on the maximum hours they can flight on a day, creating delays by flight crew unavailability.
# MAGIC 
# MAGIC There have been many researches on modeling and predicting fight delays [6], as airports and airlines would benefit significantly of predicting flight delays with such an antecedence that allowed them to re-plan their operations to minimize costs and to improve customer satisfaction. But the task is far from trivial. The development of accurate prediction models for flight delays became cumbersome due to the complexity of the air transportation system, the number of methods for prediction, and the deluge of noisy flight data [4].
# MAGIC 
# MAGIC ### - Problem statement and metric of interest
# MAGIC 
# MAGIC In this project, we adopt the airline's perspective. By early detection of a potential flight delay, an airline can re-schedule a series of flight departures and arrivals in order to optimize the use of its aircrafts, crews and airport capacity while minimizing costs related to airport costs, fuel, penalties, rebooking and accomodations.
# MAGIC 
# MAGIC The primary objective is to predict flight departure delays, where a delay is defined as a 15-minute or greater with respect to the original proposed time of departure. This is a classification problem, and for each flight we predict the label as a *delay* or a *non-delay*. The prediction is also done two hours ahead of departure with the best data available at that moment, as two hours is considered enough time for the airlines companies to arrange alternatively. 
# MAGIC 
# MAGIC As our baseline algorithm, we use Logistic Regression, and from there evolve to test Support Vector Machines. We also use two ensemble models based on classification trees: Random Forests and Gradient Boosting to compute our prediction of flight delays.
# MAGIC 
# MAGIC Since historical data suggests that relatively only a few flights are delayed, a classifier that simply predicts that no flights will be delayed can achieve remarkably low error rates [8]. We, thus, aim to improve the recall metric, which is the percentage of delayed flights that are correctly classified as delayed. Improving the recall metric is also considered as a big concern by airlines companies [8]. Previous data suggests that flight delay prediction report recalls are around 60% [9].
# MAGIC 
# MAGIC For measuring recall we apply time-series cross-validation on a rolling basis. We start with a small subset of data for training purposes, forecast the labels for the later data points and then check the accuracy for the forecasted data points. The same forecasted data points are then included as part of the next training dataset and subsequent data points are forecasted.
# MAGIC 
# MAGIC As a tie-breaking metric we will use f-Measure, because a very low precision metric would also be undesirable for our classifier.
# MAGIC 
# MAGIC ### - Hypotheses 
# MAGIC 
# MAGIC The main reasons for flight delay in US as reported by the Bureau of Transportation Statistics [7] gives us some hints about features to explore:
# MAGIC 
# MAGIC * **Frequency of delays in the departure airport (for all airlines) in the past 2, 4, 8 and 12 hours** - local issues in the departing airport may cause a series of flights from different airlines to delay (e.g. weather conditions, security incidents, etc.)
# MAGIC 
# MAGIC * **Frequency of delays in the destination airport (for all airlines) in the past 2, 4, 8 and 12 hours** - local issues in the destination airport may cause a series of flights from different airlines to delay in the origin (e.g. weather conditions, security incidents, etc.)
# MAGIC 
# MAGIC * **Frequency of delays in the most important hubs (for all airlines) in the past 2, 4, 8 and 12 hours** - local issues in important hubs of the system may cause a series of flight from different airlines in different airports to delay
# MAGIC 
# MAGIC * **Frequency of delays of the same airline (in the departure airport) in the past 2, 4, 8 and 12 hours** - majority of delays are classified as under the air carrier control, so a specific carrier previous delays might be a good indicative of operational problems that might propagate
# MAGIC 
# MAGIC * **Frequency of delays of the same airline (in the most important hubs) in the past 2, 4, 8 and 12 hours** - operational problems in airport hubs might propagate faster to the system than isolated problems
# MAGIC 
# MAGIC * **Frequency of late arrivals in the departure airport (for all airlines) in the past 2, 4, 8 and 12 hours** - late arriving aircrafts are the second most prevalent cause for flight departure delays
# MAGIC 
# MAGIC * **Frequency of late arrivals from flights in the next 2 hours or more that already departed delayed from their origins** - late arrivals from routes longer than 2 hours are a known fact that might enhance our predictions
# MAGIC 
# MAGIC * **Weather conditions at departure airport** - a basket of weather variables correlated with airport closures and flight delays
# MAGIC 
# MAGIC * **Weather conditions at destination airport** - bad weather at destination may delay takeoff at origin 
# MAGIC 
# MAGIC * **Extreme weather events** - extreme weather indicator variables in a radius of x miles from departure or destination airport

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Description of Datasets and Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### Datasets overview
# MAGIC 
# MAGIC #### Reporting Carrier On-Time Performance
# MAGIC 
# MAGIC `Source: Bureau of Transportation Statistics (BTS)`
# MAGIC 
# MAGIC This database contains scheduled and actual departure and arrival times reported by certified U.S. air carriers that account for at least one percent of domestic scheduled passenger revenues. The data is collected by the Office of Airline Information, Bureau of Transportation Statistics (BTS).
# MAGIC 
# MAGIC Reporting carriers are required to (or voluntarily) report on-time data for flights they operate: on-time arrival and departure data for non-stop domestic flights by month and year, by carrier and by origin and destination airport. Includes scheduled and actual departure and arrival times, canceled and diverted flights, taxi-out and taxi-in times, causes of delay and cancellation, air time, and non-stop distance.
# MAGIC 
# MAGIC For the first phase of the project we will focus on flights departing from two major US airports (ORD (Chicago O’Hare) and ATL (Atlanta) in the first half of 2015 (six months of data). For the final phase of the project we will focus on the entire flight data departing from all major US airports for the 2015-2019 timeframe. It is dataframe of shape 31,746,841 x 109.
# MAGIC 
# MAGIC #### Integrated Surface Data (ISD)
# MAGIC 
# MAGIC `Source: Federal Climate Complex (FCC) in Asheville, NC`
# MAGIC 
# MAGIC The Integrated Surface Data (ISD) is composed of worldwide surface weather observations from over 20,000 stations, collected and stored from sources such as the Automated Weather Network (AWN), the Global Telecommunications System (GTS), the Automated Surface Observing System (ASOS), and data keyed from paper forms. Most digital observations are decoded either at operational centers and forwarded to the Federal Climate Complex (FCC) in Asheville, NC, or decoded at the FCC. NOAA’s National Centers for Enviornmental Information (NCEI) and the US Air Force’s 14th Weather Squadron (14WS) make up the FCC in Asheville, NC.
# MAGIC 
# MAGIC The dataset is used in climatological applications by numerous DOD and civilian customers. The dataset includes data originating from various codes such as synoptic, airways, METAR (Meteorological Routine Weather Report), and SMARS (Supplementary Marine Reporting Station), as well as observations from automatic weather stations. The data are sorted by station, year-month-day-hourminute, report type, and data source flag.
# MAGIC 
# MAGIC The weather data available to the project is for the period of Jan 2015 to December 2019. It is a dataframe of shape 630,904,436 x 177.
# MAGIC 
# MAGIC #### Aviation Support Tables - Master Coordinate
# MAGIC 
# MAGIC `Source: Bureau of Transportation Statistics (BTS)`
# MAGIC 
# MAGIC The Aviation Support Tables provide comprehensive information about U.S. and foreign air carriers, carrier entities, worldwide airport locations, and other geographic data. This information is developed and maintained by the Office of Airline Information, Bureau of Transportation Statistics, and is updated on an on-going basis. Data include U.S. Carriers and foreign carriers with at least one point of service in the United States or one of its territories.
# MAGIC 
# MAGIC The Master Coordinate table contains historical (time-based) information on airports used throughout the aviation databases. It provides a list of domestic and foreign airport codes and their associated world area code, country information, state information (if applicable), city name, airport name, city market information, and latitude and longitude information.
# MAGIC 
# MAGIC It is a dataframe of shape 18,097 x 10.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploratory Data Analysis
# MAGIC 
# MAGIC The following links point to notebooks with the full description of the EDA performed in each of the datasets.
# MAGIC 
# MAGIC - <a href="$./starter_nb_fp_KM_v2">Flight's on-time performance data</a> 
# MAGIC 
# MAGIC - <a href="$./starter_nb_fp_KF_v9">EDA in the Weather Dataset</a>
# MAGIC 
# MAGIC - Integrated Surface Data (ISD)
# MAGIC 
# MAGIC - Aviation Support Tables - Master Coordinate
# MAGIC 
# MAGIC -  <a href="$./starter_nb_fp_AD_v4">Joined Dataset of the weather and airlines data</a>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key Challenges
# MAGIC 
# MAGIC * Getting the closest weather station to each airport - 
# MAGIC <p> The airlines data table has a weather station column, but the name of the weather station doesnot always cooresspond to the weather stations in the weather table. To solve this problem we adopted a multistep methodology. First we extracted the latitude and longitude of each departing airport city using Google's geocode api. We map back these coordinates to a matching weather station latitude and longitude using Haversine formula. The Havesine formula is used to determine the great-circle distance between two points on a sphere given their longitudes and latitudes. Now each departing city in the weather data has a weather station from where we can extract the hourly weather data.
# MAGIC 
# MAGIC * Unbalanced dataset - There is a huge imbalance of data in our working dataset. For every delayed flight there are 4.3 non-delayed ones. This kind of imbalance in the training data is expected to harm the ability of our classifiers to learn how to detect a delayed flight. In fact, our classifiers might have a tendency to say every flight is not-delayed and still achieve high accuracy. To counterbalance that we oversampled the minority class (delayed flights), the the details of which can be found <a href=“$./fp_algo_implem_v2”>in this notebook</a>.
# MAGIC 
# MAGIC * Missing values - For flight performance data we observed that there were several columns (especially related to diverted flights) that had high number of missing values. We took a high threshold of 96% and any columns with over 96% of the total values missing were removed from the dataset. Originally the dataset had 109 columns and after removing there were 61 fields remaining. @Kevin
# MAGIC   
# MAGIC * Size of the feature space - Flight performance data had originally 63493682 rows and 109 fields. However there wee duplicate rows in the dataset and after reoving duplicates the final count was 31746841 rows. Further details on EDA of flight data are in the notebook linked to flight perforamnce data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature Creation

# COMMAND ----------

# MAGIC %md
# MAGIC ### <a href="$./starter_nb_fp_KM_v2">Flight data</a>
# MAGIC 
# MAGIC Several colums were added to the flight data to fascilitate join with weather dataset and based on observations in the flight dataset on EDA. Also studying the main reasons for flight delay in US as reported by the Bureau of Transportation Statistics [7] we got some hints about features to explore and engineer. We added a timestamp feature that included scheduled date and time of flight departure, to fasciliate join with weather data. We added more features related to frequency of delays in the departure airport, arrival airport, specific airlines, important hubs and frequency of late arrivals in departure airport at different time points before the prediction time (2 hours prior to scheduled flight departure time). We also added average delayed flights per aircraft to account for aging and faulty aircrafts and part of the day (Morning, Afternoon etc.) the flight was scheduled to depart. We further calculated and added percentage flights with weather delays to fascilitate feature engineering using weather data. Further details on <a href="$./starter_nb_fp_KM_v2"> feature engineering and creation for flights data can be found in this notebook.</a>
# MAGIC 
# MAGIC ### <a href="$./starter_nb_fp_KF_v9">Weather Data</a>
# MAGIC 
# MAGIC There was one feature that was created in the weather data that aimed to predict if, given the weather data at the time, the weather at that current location was good or bad. We implemented many filters and constraints as to what defines/constitutes "bad" or "good" weather. In addition, many of the features had to be scaled down by a factor of 10, as well as cleaned and casted from strings into floats. This is all elaborated upon in <a href="$./starter_nb_fp_KF_v9"> this specific notebook </a>. 
# MAGIC 
# MAGIC ### <a href="$./starter_nb_fp_AD_v5">Joined Data</a> 
# MAGIC 
# MAGIC Two columns were added to the joined dataframe - a row number that treats each departed flight entry as unique and the weather station nearest to each departing airport.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Algorithm Exploration & Implementation

# COMMAND ----------

# MAGIC %md
# MAGIC In this section we load the data from our check point after EDA and build a pipeline for training and evaluating Logistic Regression, Random Forest, SVM and Gradient Boosting Tree using Time Series Cross Validation on a rolling basis (a custom class we built at pyspark). We conclude our best model is GBT, followed by RF. <a href='$./fp_algo_exploration_v2'>The notebook for this implementation can be found here</a>.
# MAGIC 
# MAGIC We next explain the math behind Regularized Logistic Regression with Elastic Net and implements it from scratch in a distributed way using Spark RDD.
# MAGIC We apply the algorithm to a small sample of the data (3 months) and then compare the results with the Logistic Regression method available at the pyspark.ml library.
# MAGIC <a href='$./fp_algo_implem_v2'>The notebook for this implementation can be found here</a>.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Results and Discussions

# COMMAND ----------

# MAGIC %md
# MAGIC On our main metric of interest (Recall), all algorithms but Logistic Regression had remarkable performances, with Recall above 94%. In terms of fMeasure (out tie-break metric), Gradient Boosting Tree had a slightly better performance, due to its higher Precision compared to Random Forest and SVM. So the winner is GBT. GBT took the largest time to train, but since we don't antecipate the need for constant retraining this should not be a concern. Prediction time on the other hand takes slighly longer on ensemble models like GBT than on not-emsemble models such as SVM or Logistic Regression. If performance at prediction time is important, this consideration could make us opt for SVM instead, which is as strong as in terms of Recall but slightly less performant in terms of fMeasure and Precision.
# MAGIC 
# MAGIC Our own Logistic Regression implementation had a worst performance when compared to the pyspark.ml.classification.LogisticRegression implementation. We got a delay recall of 0.43 (vs 0.68 of pyspark implementation) and a delay precision of 0.29 (vs. 0.37 of pyspark implementation). We tried several adjustments to match the results but the differences continued to be large. We tried to keep the parameter sets the most comparable we could but there are many more parameters in the LogisticRegression class than in our own implementation. We are not sure as well that the pyspark implementation do use Batch Gradient Descent as the optimization algorithm under the hood. If it uses other algorithms (e.g. SGD or LBFGS) than it is understandable that results will be less comparable. The learning rate parameter is also a parameter that is not transparent in the pyspark implementation.
# MAGIC 
# MAGIC Our work is more detailed in the notebook found <a href='$./fp_algo_implem_v2'>here</a>.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Challenges faced and future work
# MAGIC 
# MAGIC We faced several challenges over the course of this project which were not just limited to cluster inefficiencies and dearth and time.
# MAGIC - Joining flight and weather data: Flight data lacked timestamps and had local time. We had to calculate local timezones to create timestamps, and convert them further to utc and unix to fasciliate join.
# MAGIC - A common column between the two data tables to join didnot exist. The airlines data table has a weather station column, but the name of the weather station doesnot always cooresspond to the weather stations in the weather table.
# MAGIC - The joined dataframe gets saved as a parquet file without any complaints but when we try to use this as acheckpoint and reload it on another notebook for further analysis, the count of the data frame is drastically lower.
# MAGIC 
# MAGIC There were several places where this project could have been improved if we had appropriate time and resources. For handling oversampling of data, we could have implemented the ROSE function (Random Over-Sampling Examples) as available withing pyspark. The other alternative was to use the SMOTE or Synthetic Minority Oversampling Technique that also addresses imbalanced datasets to oversample the minority class.
# MAGIC 
# MAGIC We were unlucky in not being able to work with the full dataset due to unexplainable technical difficulties but we made sure that our data (of 2 million rows) was a good representative sample of the total population of 26 million rows. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Conclusions
# MAGIC 
# MAGIC On our main metric of interest (Recall), the best-performing algorithms were Random Forest and GBT, with Recall slightly above 80%. In terms of fMeasure (out tie-break metric), Gradient Boosting Tree had a slightly better performance, due to its higher Precision compared to Random Forest. So the winner is GBT. GBT is expensive to train, but since we don't antecipate the need for constant retraining this should not be a concern. Prediction time on the other hand takes slighly longer on ensemble models like GBT or RF than on not-emsemble models such as SVM or Logistic Regression. If performance at prediction time is important, this consideration could make us opt for SVM instead, which is as strong as in terms of Recall but slightly less performant in terms of fMeasure and Precision.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Application of Course Concepts
# MAGIC 
# MAGIC The following course concepts were implemented in our project. 
# MAGIC 
# MAGIC - All our data sets were imported in the form of SPARK dataframes and we used Spark's inbuilt ML library for all algorithms' implementation. 
# MAGIC   - Since the weather dataset is orders of magnitude larger than the airlines dataset, EDA/analysis could not be done with Pandas Dataframes. Instead, we had to perform analysis while working within the Pyspark Dataframes. Due to parallel execution on all cores on multiple machines, PySpark runs operations faster than pandas [10]. As we learned in class, pandas DataFrames run operations on a single node whereas PySpark runs on multiple machines [10]. Within the Pyspark Dataframes, our preferred choice was to write code that processes the data in a parallel fashion, so both the time cost and compute cost would be minimized. However, in some cases, if the code did not exist or was incredibly complex to reproduce in Pyspark Dataframes, we had to process it in our non-preferred method in a non-parallel fashion (eg. correlation matrix creation had to be performed in Pandas). 
# MAGIC - Our baseline algorithm is Regularized Logistic Regression with Elastic Net and we implemented it from scratch in a distributed way using Spark RDD.
# MAGIC - We implement the Random Forest, Gradient Boosted Trees and Support Vector Machines, the concepts of which we leart in class.
# MAGIC - This is a full implementation of Spark on a cluster setup.
# MAGIC - We also read and wrote our files in a parquet format since it is 1) more Spark friendly 2) memory efficient (1/20th the size of a CSV file), and 3) very fast in terms of processing.

# COMMAND ----------

# MAGIC %md
# MAGIC Citations:
# MAGIC 
# MAGIC [1] Thiagarajan B, et al. A machine learning approach for prediction of on-time performance of flights. In 2017 IEEE/AIAA 36th Digital Avionics Systems Conference (DASC). New York: IEEE. 2017.
# MAGIC 
# MAGIC [2] Eurocontrol Network Manager Annual Report. 2019. Available online: https://www.eurocontrol.int/publication/networkmanager-annual-report-2019.
# MAGIC 
# MAGIC [3] Peterson, Everett B., Kevin Neels, Nathan Barczi, and Thea Graham. The Economic Cost of Airline Flight Delay. In Journal of Transport Economics and Policy 47, no. 1 (2013): pages 107–121. http://www.jstor.org/stable/24396355.
# MAGIC 
# MAGIC [4] Carvalho, Leonardo and Sternberg, Alice and Maia Gonçalves, Leandro and Beatriz Cruz, Ana and Soares, Jorge A. and Brandão, Diego and Carvalho, Diego and Ogasawara, Eduardo. On the relevance of data science for flight delay research: a systematic review. In Transport Reviews, volume 41, pages 499–528, 2020.
# MAGIC 
# MAGIC [5] Rome, J.A., S.D. Rose, R.W. Lee, J.H. Cistone, G.F. Bell, and W.S. Leber. Ripple delay and its mitigation. In Air Traffic Control Quarterly, Vol. 9(2), pp. 59-98, 2001.
# MAGIC 
# MAGIC [6] Yazdi, M.F., Kamel, S.R., Chabok, S.J.M. et al. Flight delay prediction based on deep learning and Levenberg-Marquart algorithm. J Big Data 7, 106 (2020). https://doi.org/10.1186/s40537-020-00380-z
# MAGIC 
# MAGIC [7] Bureau of Transportation Statistics. Understanding the Reporting of Causes of Flight Delays and Cancellations. Available online: https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations
# MAGIC     
# MAGIC [8] Lawson, Dieterich and Castillo, William. Predicting Flight Delays. http://cs229.stanford.edu/proj2012/CastilloLawson-PredictingFlightDelays.pdf
# MAGIC 
# MAGIC [9] http://www.datawrangling.com/how­flightcaster­squeezes­predictions­from­flight­data
# MAGIC 
# MAGIC [10] https://sparkbyexamples.com/pyspark-tutorial/#pyspark-graphframes

# COMMAND ----------


