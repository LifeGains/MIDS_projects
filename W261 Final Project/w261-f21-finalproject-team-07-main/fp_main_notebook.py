# Databricks notebook source
# MAGIC %md
# MAGIC # Final Project: Flight Delay Prediction
# MAGIC `MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Fall 2021`
# MAGIC 
# MAGIC **`Team 07: Atreyi Dasmahapatra, Kanika Mahajan, Kevin Fu, Lucas Barbosa`**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table of Contents
# MAGIC 
# MAGIC 1. Question Formulation
# MAGIC 
# MAGIC 3. EDA & Discussion of Challenges
# MAGIC 
# MAGIC 4. Feature Engineering
# MAGIC 
# MAGIC 5. Algorithm Exploration
# MAGIC 
# MAGIC 6. Algorithm Implementation
# MAGIC 
# MAGIC 7. Conclusions
# MAGIC 
# MAGIC 8. Application of Course Concepts

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 1. Question Formulation

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Why does flight delay prediction matter?
# MAGIC 
# MAGIC Flight delays are inevitable but still a curse of modern life, causing billions of dolars of extra costs for airlines and passengers every year. In 2017, only 76% of the flights in US arrived on time [1]. During the period between 2013 and 2019, while the number of flights in Europe increased by 16%, the average departure delay of European flights increased by 41% [2]. Delays cost Americans an estimated $28 billion in 2018, including direct and indirect costs to airlines and passengers [3].
# MAGIC 
# MAGIC Delays do impact airlines and passengers alike. Passengers incur in extra time to get at their destinations, increasing their trip costs [4], not to mention the inconvenience of lenghty stays at airports and missed connections. The airlines incur in penalties, fines and additional operation costs, such as crew and aircrafts retentions in airports [4]. Delays also jeopardize airlines marketing strategies [4], since carriers rely on customers’ loyalty to support their growth and financial standing and the consumer’s choice is significantly affected by unreliable performance.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### What does cause a flight to delay?
# MAGIC 
# MAGIC Since 2003, airlines operating in the US need to report the causes of delay in broad categories that were created by the Air Carrier On-Time Reporting Advisory Committee [7]. The categories are:
# MAGIC 
# MAGIC * **Air Carrier:** The cause of the cancellation or delay was due to circumstances within the airline's control (e.g. maintenance or crew problems, aircraft cleaning, baggage loading, fueling, etc.).
# MAGIC * **Late-arriving aircraft:** A previous flight with same aircraft arrived late, causing the present flight to depart late.
# MAGIC * **National Aviation System (NAS):** Delays and cancellations attributable to the national aviation system that refer to a broad set of conditions, such as non-extreme weather conditions, airport operations, heavy traffic volume, and air traffic control.
# MAGIC * **Extreme Weather:** Significant meteorological conditions (actual or forecasted) that, in the judgment of the carrier, delays or prevents the operation of a flight such as tornado, blizzard or hurricane.
# MAGIC * **Security:** Delays or cancellations caused by evacuation of a terminal or concourse, re-boarding of aircraft because of security breach, inoperative screening equipment and/or long lines in excess of 29 minutes at screening areas.
# MAGIC 
# MAGIC Data reported by the Bureau of Transportation Statistics shows that in 2020 the most commom causes for delay in US were Air Carrier delay (41%), Aircraft arriving late (30.2%), NAS Delay (21.6%), Extreme Weather (6.9%) and Security (0.2%) [7].

# COMMAND ----------

# MAGIC %md
# MAGIC ### Why is it difficult to predict?
# MAGIC 
# MAGIC Commercial aviation is a complex transportation system. It deals with a sophisticated and interconnected origin-destination network that needs orchestration to provide smooth operations [4]. Every day, airlines do plan various schedules for aircrafts, pilots and flight attendants. Delays have ripple effects, where delays in one part of the network quickly propagate to the others. Turn times between flight legs are constantly minimized by airlines in order to fully utilize expensive resources, so there is not much slack in the schedule to absorb delays [5]. Crews are also commonly shared between flight legs [6] and have restrictions on the maximum hours they can flight on a day, creating delays by flight crew unavailability.
# MAGIC 
# MAGIC There have been many researches on modeling and predicting fight delays [6], as airports and airlines would benefit significantly of predicting flight delays with such an antecedence that allowed them to re-plan their operations to minimize costs and to improve customer satisfaction. But the task is far from trivial. The development of accurate prediction models for flight delays became cumbersome due to the complexity of the air transportation system, the number of methods for prediction, and the deluge of noisy flight data [4].

# COMMAND ----------

# MAGIC %md
# MAGIC ### Problem definition and metric of interest
# MAGIC 
# MAGIC In this project, we will adopt the airline's perspective. Flight delay prediction is important to airlines as it can inform several actions related to airline delay management. By early detecting a potential delay, an airline can re-schedule a series of flight departures and arrivals in order to optimize the use of its aircrafts, crews and airport capacity while minimizing costs related to airport costs, fuel, penalties, rebooking, accomodation, among others.
# MAGIC 
# MAGIC Our goal is to predict flight departure delays, where a delay is defined as a 15-minute or greater delay with respect to the planned time of departure. We will treat the problem as a classification problem, where for each flight we predict a delay or a non-delay. We will predict any delays, irrespective of the root causes. The prediction will be done two hours ahead of departure with the best data available at that moment.
# MAGIC 
# MAGIC We will implement Logistic Regression as our baseline algorithm, and from there evolve to test SVM and two ensemble models based on classification trees: Random Forests and Gradient Boosting.
# MAGIC 
# MAGIC Since relatively few flights are delayed, a classifier that simply predicts that no flights will be delayed can achieve remarkably low error rates [8]. Thus, our main challenge is to instead improve recall, the percentage of delayed flights that are correctly classified as delayed. Companies in industry that work in flight delay prediction report recalls around 60% [9]. These companies also cite improving recall as their biggest concern [8].
# MAGIC 
# MAGIC For measuring recall we will apply time-series cross-validation on a rolling basis. We will start with a small subset of data for training purpose, forecast for the later data points and then checking the accuracy for the forecasted data points. The same forecasted data points are then included as part of the next training dataset and subsequent data points are forecasted.
# MAGIC 
# MAGIC As a tie-break metric we will use fMeasure, because too low precision would also be undesirable for our classifier.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Features hypotheses
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
# MAGIC 
# MAGIC * ...

# COMMAND ----------

# MAGIC %md
# MAGIC 
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. EDA & Discussion of Challenges

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
# MAGIC Reporting Carrier On-Time Performance
# MAGIC 
# MAGIC Integrated Surface Data (ISD)
# MAGIC 
# MAGIC Aviation Support Tables - Master Coordinate

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary of the key challenges
# MAGIC 
# MAGIC * Getting the closest weather station to each airport
# MAGIC * Unbalance of data
# MAGIC * Missing values
# MAGIC * Size of the feature space

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Weather Dataset Challenges:**
# MAGIC 
# MAGIC First, we have a very large dataset of 630 million rows. This implies lots of compute and time cost embedded in analyzing/shuffling this large amount of data. Some parts of the EDA such as evaluating the number of nulls or empty rows/columns had to be done and generalized with the smaller dataset (3 months) as opposed to the full dataset because the full dataset would generate an out of memory error.
# MAGIC 
# MAGIC Before any analysis could be done, there were many missing/erroneous features that we must remove. 161 out of 177 features had, on average, 50%+ missing data points (anywhere from 15-29 million missing rows out of 29,823,926 total in the 3 month data set), so we discarded those features. Next, since we were working within the US, we excluded any weather station/reading that was located outside of the US. There were 19 features that needed to be un-nested from 6 columns. Out of those 19 features, there were missing/erroneous rows that needed to be removed (coded as 999, 9999, or various iterations of 9’s). There were also quality codes that indicated that the value recorded was inaccurate (<5% of overall dataset); hence we discarded those rows as well. After additional cleaning (removing symbols), scaling (by a factor of 10), and casting, 42 million rows remained. We did not proceed with imputing missing/erroneous variables at this stage because we still have an abundance of data (42 million rows) for our model to train on.
# MAGIC 
# MAGIC **Other EDA steps included:**
# MAGIC 
# MAGIC We determined that there were many columns (such as quality codes or wind angle) that were simply identification features as opposed to predictive features.
# MAGIC 	Using a correlation matrix, we found that air temperature and dew point temperature have a high positive correlation of 0.8. All other significant variables were relatively uncorrelated with each other (uncorrelated defined as <|0.3|).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC xxxx

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Weather Dataset**
# MAGIC 
# MAGIC We created a binary feature called “Bad_Weather_Prediction.” After discarding the missing/erroneous rows in the usable feature set, we were left with a total of 42 million rows. First, we established an upper limit. The max number of days that it rains per year in the US according to this source is 45% of all days. Therefore, our predictor should not exceed 18.9 million bad weather data point predictions (42 x 0.45), since that would assume the max amount of rainfall in every state.
# MAGIC 
# MAGIC We focused on the air and dew temperatures first. When air temperature and dew point temperatures are very close, the air has a high relative humidity and a lower chance of rain. When there is a large difference between air and dew point temperatures, this indicates lower relative humidity and a higher chance of rain (calculated with this formula here). These sources (1, 2) state that a reading above 50% elevates the chance of rain. We thought this was a basic filter that predicted whether bad weather was coming, and more filters can be added on top of this to make it more robust.
# MAGIC Next, we developed three additional criteria in which one of three would need to be met in order to firmly predict that bad weather is indeed on its way. Wind speeds in excess of 34 mph (15.2 meters per second) is indicative of bad weather (source). Low sea pressure (source) also indicates bad weather, so we defined “low sea pressure” as any reading in the bottom 10% of the dataset. The required visibility required for a plane to take off is 1 mile or 1609.34 meters (source). Therefore, any reading below 1609.34 meters is considered a bad weather indicator as well.
# MAGIC Additional considerations include considering below freezing (0 degrees Celsius) as bad weather, since below freezing indicates a higher chance of ice and snow on the runways. However, we felt that this was too broad of an interpretation and showed many false positives in colder areas such as Alaska and states in the Northeast. We also looked into the minimum ceiling height dimension required: “the means of egress shall have a ceiling height of not less than 7 feet 6 inches (2286 mm or 2.286 meters)” (source). However, the rows in our dataset show a very bifurcated reading with the 50th percentile at 22,000 mm and the other half clumped around <4000 mm. We concluded that using this feature would not be as informative or predictive as the other criteria if half of the dataset (21 million) would be labeled as “bad weather.” This would also violate our upper bound assumption of 18.9 million rows.
# MAGIC 
# MAGIC With the four criteria mentioned above, the Bad_Weather_Prediction column was added: 1 for bad weather, 0 for good weather. We netted a total of 3,956,117 rows (~9.4% of the total dataset). Though not verified with rigorous scientific precision, this makes logical sense to us since we can reasonably conclude that bad weather in the US, on average, occurs around 10-15% of the time, taking into account the difference between wet and dry states, increase in global warming, etc.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Algorithm Exploration

# COMMAND ----------

# MAGIC %md
# MAGIC In this section we load the data from our check point after EDA and build a pipeline for training and evaluating Logistic Regression, Random Forest, SVM and Gradient Boosting Tree using Time Series Cross Validation on a rolling basis (a custom class we built at pyspark). We conclude our best model is GBT, followed by SVM. <a href='$./fp_algo_exploration'>The notebook for this implementation can be found here</a>.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Algorithm Implementation

# COMMAND ----------

# MAGIC %md
# MAGIC In this section we explain the math behind Regularized Logistic Regression with Elastic Net and implements it from scratch in a distributed way using Spark RDD.
# MAGIC We apply the algorithm to a small sample of the data (3 months) and then compare the results with the Logistic Regression method available at the pyspark.ml library.
# MAGIC <a href='$./fp_algo_implem'>The notebook for this implementation can be found here</a>..

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Conclusions

# COMMAND ----------

# MAGIC %md
# MAGIC On our main metric of interest (Recall), all algorithms but Logistic Regression had remarkable performances, with Recall above 94%. In terms of fMeasure (out tie-break metric), Gradient Boosting Tree had a slightly better performance, due to its higher Precision compared to Random Forest and SVM. So the winner is GBT. GBT took the largest time to train, but since we don't antecipate the need for constant retraining this should not be a concern. Prediction time on the other hand takes slighly longer on ensemble models like GBT than on not-emsemble models such as SVM or Logistic Regression. If performance at prediction time is important, this consideration could make us opt for SVM instead, which is as strong as in terms of Recall but slightly less performant in terms of fMeasure and Precision.
# MAGIC 
# MAGIC Our own Logistic Regression implementation had a worst performance when compared to the pyspark.ml.classification.LogisticRegression implementation. We got a delay recall of 0.43 (vs 0.65 of pyspark implementation) and a delay precision of 0.29 (vs. 0.57 of pyspark implementation). We tried several adjustments to match the results but the differences continued to be large. We tried to keep the parameter sets the most comparable we could but there are many more parameters in the LogisticRegression class than in our own implementation. We are not sure as well that the pyspark implementation do use Batch Gradient Descent as the optimization algorithm under the hood. If it uses other algorithms (e.g. SGD or LBFGS) than it is understandable that results will be less comparable. The learning rate parameter is also a parameter that is not transparent in the pyspark implementation.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Application of Course Concepts

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Since the weather dataset is orders of magnitude larger than the airlines dataset, EDA/analysis could not be done with Pandas Dataframes. Instead, we had to perform analysis while working within the Pyspark Dataframes. Due to parallel execution on all cores on multiple machines, PySpark runs operations faster than pandas (Source). As we learned in class, pandas DataFrames run operations on a single node whereas PySpark runs on multiple machines (Source). Within the Pyspark Dataframes, our preferred choice was to write code that processes the data in a parallel fashion, so both the time cost and compute cost would be minimized. However, in some cases, if the code did not exist or was incredibly complex to reproduce in Pyspark Dataframes, we had to process it in our non-preferred method in a non-parallel fashion (eg. correlation matrix creation had to be performed in Pandas).
# MAGIC 
# MAGIC 
# MAGIC - Running in parallel/distributed systems
# MAGIC - Pagerank of airports
# MAGIC - Cross validation
# MAGIC - Logistic regression
# MAGIC 
# MAGIC Given starter topic list from Google Doc --
# MAGIC 
# MAGIC scalability / time complexity / I/O vs Memory
# MAGIC functional programming / higher order functions / map reduce paradigm
# MAGIC bias variance tradeoff / model complexity / regularization
# MAGIC associative/commutative operations
# MAGIC race conditions / barrier synchronization / statelessness
# MAGIC the shuffle  / combiners / local aggregation
# MAGIC order inversion pattern / composite keys 
# MAGIC total order sort /  custom partitioning
# MAGIC broadcasting / caching / DAGs / lazy evaluation
# MAGIC GD - convex optimization / Batch vs stochastic
# MAGIC sparse representation (pairs vs stripes)
# MAGIC preserving the graph structure / additional payloads 
# MAGIC One Hot Encoding / vector embeddings / feature selection
# MAGIC normalization
# MAGIC assumptions (for different algorithms - for example OLS vs Trees)
# MAGIC implications of iterative algorithms in Hadoop M/R vs Spark

# COMMAND ----------


