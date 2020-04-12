# Databricks notebook source
# MAGIC %md <h1 align="center"> Airline Delays </h1>
# MAGIC <h2 align="center"> W261 - Final Project </h2>
# MAGIC <h5 align="center"> by Adom Sohn, Chandra Shekar Bikkanur, Jayesh Parikh, Tucker Anderson</h5>

# COMMAND ----------

# MAGIC %md <h2 align="center"> Introduction:</h2>
# MAGIC 
# MAGIC In this project, we have gathered airline data across United States and corresponding weather data to predict the arrival delay of a flight in minutes.

# COMMAND ----------

# MAGIC %md <h2 align="center"> Libraries:</h2>
# MAGIC 
# MAGIC We will first load the required Python, PySpark and other required libraries to run the analysis.

# COMMAND ----------

import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import functools
import dateutil.parser
import datetime
from math import atan2, cos, sin, radians, degrees

from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType, TimestampType
from pyspark.sql import SQLContext
from pyspark.sql.functions import col, concat, lit, udf
from pyspark.sql import DataFrameNaFunctions
sqlContext = SQLContext(sc)

from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator, CrossValidatorModel
from pyspark.ml.stat import ChiSquareTest

# COMMAND ----------

# MAGIC %md <h2 align="center">Data:</h2>
# MAGIC 
# MAGIC For this analysis, we are going to import airlines data, weather data and airport codes data across United States.

# COMMAND ----------

#Read in airlines, weather, stations, airport codes dataset
airlines = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/201*.parquet")
weather_parquet = spark.read.option("header", "true")\
                      .parquet(f"dbfs:/mnt/mids-w261/data/datasets_final_project/new_weather_parquet_177/weather201*a.parquet")
stations = spark.read.option("header", "true").csv("dbfs:/mnt/mids-w261/data/DEMO8/gsod/stations.csv.gz")
airport_codes = spark.read.csv('/FileStore/tables/airport_codes.csv', header="true", inferSchema="true")
airport_codes = airport_codes.selectExpr("`IATA Code` as code", "Latitude as lat", "Longitude as lon")

# COMMAND ----------

# MAGIC %md <h2 align="center">Airlines & Weather Data Merge:</h2>
# MAGIC 
# MAGIC We will merge airlines and weather data sets to form a composite dataset.

# COMMAND ----------

def is_Weekend(x):
  """
  Function to determine if a given day of the week is a weekend_day(Friday, Saturday, Sunday)
  """
  if   x < 5: 
    return 0
  else: 
    return 1

def is_RushHour(x):
  """
  Function to determine if a given time of the day is rush hour (1600-2100)
  """
  if (x != None) and (x >= 1600) and (x <= 2100): 
    return 1
  else: 
    return 0
 
def preprocessAirlines(df):
  cols_to_keep = ['MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'DEP_DELAY', 'DEP_TIME_BLK', 'ARR_DELAY', 'ARR_TIME_BLK', 'CRS_ELAPSED_TIME', 'DISTANCE',  'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'IS_WEEKEND', 'DEP_RUSH_HOUR', 'ARR_RUSH_HOUR', 'DEP_TIME', 'CRS_DEP_TIME', 'ARR_TIME', 'CRS_ARR_TIME']
  cols_to_remove = [x for x in df.columns if x not in cols_to_keep]
  df = df.orderBy("FL_DATE") 
  df = df.filter(df.CANCELLED == False)
  df = df.filter(df.DIVERTED == False)
  df = df.withColumn('CARRIER_DELAY', f.when(df.CARRIER_DELAY.isNotNull(), 1).otherwise(0))
  df = df.withColumn('WEATHER_DELAY', f.when(df.WEATHER_DELAY.isNotNull(), 1).otherwise(0))
  df = df.withColumn('NAS_DELAY', f.when(df.NAS_DELAY.isNotNull(), 1).otherwise(0))
  df = df.withColumn('SECURITY_DELAY', f.when(df.SECURITY_DELAY.isNotNull(), 1).otherwise(0))
  df = df.withColumn('LATE_AIRCRAFT_DELAY', f.when(df.LATE_AIRCRAFT_DELAY.isNotNull(), 1).otherwise(0))
  df = df.withColumn("IS_WEEKEND", f.udf(is_Weekend, IntegerType())("DAY_OF_WEEK"))
  df = df.withColumn("DEP_RUSH_HOUR", f.udf(is_RushHour, IntegerType())("DEP_TIME"))
  df = df.withColumn("ARR_RUSH_HOUR", f.udf(is_RushHour, IntegerType())("CRS_ARR_TIME"))
  df = df.fillna(0, subset=['ARR_DELAY', 'DEP_DELAY'])  
  df = df.withColumn('ORIGIN_CARRIER', concat(col("ORIGIN"), lit("_"), col("OP_UNIQUE_CARRIER")))
  df = df.withColumn('DEST_CARRIER', concat(col("DEST"), lit("_"), col("OP_UNIQUE_CARRIER")))
  preprocessAirlines_df = df.drop(*cols_to_remove)
  return preprocessAirlines_df

# COMMAND ----------

def unionAll_fn(dfs):
    return functools.reduce(lambda df1,df2: df1.union(df2.select(df1.columns)), dfs) 

def US_fn(df):
    """
    Reduce df to US only to reduce size of dataset
    """
    # US is lat/long ranges according to format: [[(lat_low, lat_high),(long_low, long_high)], [(lat_low, lat_high),(long_low, long_high)]]
    US = [[(24,49),(-125,-67)],[(17,19),(-68,-65.5)], [(13,14),(144,145)], [(15,16),(145,146)], [(-15,-14), (-171,-170)], [(18,19),(-65.4,-64)], [(18,23),(-160,-154)], [(50,175),(-170,-103)]]  

    list_df = [] #empty list for parquet parts
    parquet_part = spark.range(0).drop("id") #empty spark df

    #Filtering for individual areas in US
    for item in US:
      parquet_part = df.filter((f.col('Latitude') > item[0][0]) & (f.col('Latitude') < item[0][1]) & (f.col('Longitude') > item[1][0]) & (f.col('Longitude') < item[1][1]))
      list_df.append(parquet_part)
    
    #Appending each individual US area
    parquet_us = unionAll_fn(list_df)

    return parquet_us

def reduce_split_cols_fn(weather_parquet_us):
    """
    Reduce weather dataset to columns of interest and return split columns with comma-separated values into multiple columns for each comma-separated value.
    """
    #Reduce weather dataset to columns of interest (high level) and return split columns with comma-separated values into multiple columns for each comma-separated value.
    weather_pre_split = weather_parquet_us.select('STATION','DATE','SOURCE','LATITUDE','LONGITUDE',f.split('WND', ',').alias('WND'),f.split('VIS', ',').alias('VIS'),f.split('SLP', ',').alias('SLP'),f.split('AA1', ',').alias('AA1'))
    df_sizes_WND = weather_pre_split.select(f.size('WND').alias('WND'))
    df_sizes_VIS = weather_pre_split.select(f.size('VIS').alias('VIS'))
    df_sizes_SLP = weather_pre_split.select(f.size('SLP').alias('SLP'))
    df_sizes_AA1 = weather_pre_split.select(f.size('AA1').alias('AA1'))
    df_max_WND = df_sizes_WND.agg(f.max('WND'))
    df_max_VIS = df_sizes_VIS.agg(f.max('VIS'))
    df_max_SLP = df_sizes_SLP.agg(f.max('SLP'))
    df_max_AA1 = df_sizes_AA1.agg(f.max('AA1'))
    nb_columns_WND = df_max_WND.collect()[0][0]
    nb_columns_VIS = df_max_VIS.collect()[0][0]
    nb_columns_SLP = df_max_SLP.collect()[0][0]
    nb_columns_AA1 = df_max_AA1.collect()[0][0]
    weather_post_split = weather_pre_split.select('STATION','DATE','SOURCE','LATITUDE','LONGITUDE',*[weather_pre_split['WND'][i] for i in range(nb_columns_WND)],*[weather_pre_split['VIS'][i] for i in range(nb_columns_VIS)],*[weather_pre_split['SLP'][i] for i in range(nb_columns_SLP)],*[weather_pre_split['AA1'][i] for i in range(nb_columns_AA1)])
  
    #Filtering out data with quality issues. All string values are indicative of quality issue
    fltr_msk = [
    f.col('WND[0]') != '999',
    f.col('WND[1]') != '2',
    f.col('WND[1]') != '3',
    f.col('WND[1]') != '6',
    f.col('WND[1]') != '7',
    f.col('WND[2]') != '9',
    f.col('WND[3]') != '9999',  
    f.col('WND[4]') != '2',
    f.col('WND[4]') != '3',
    f.col('WND[4]') != '6',
    f.col('WND[4]') != '7',
    f.col('VIS[0]') != '999999',
    f.col('VIS[1]') != '2',
    f.col('VIS[1]') != '3',
    f.col('VIS[1]') != '6',
    f.col('VIS[1]') != '7',
    f.col('VIS[2]') != '9',
    f.col('VIS[3]') != '2',
    f.col('VIS[3]') != '3',
    f.col('VIS[3]') != '6',
    f.col('VIS[3]') != '7',
    f.col('SLP[0]') != '99999',
    f.col('SLP[1]') != '2',
    f.col('SLP[1]') != '3',
    f.col('SLP[1]') != '6',
    f.col('SLP[1]') != '7',
    f.col('SLP[1]') != '9',
    f.col('AA1[0]') != '99',
    f.col('AA1[1]') != '9999',
    f.col('AA1[2]') != '9',
    f.col('AA1[3]') != '2',
    f.col('AA1[3]') != '3',
    f.col('AA1[3]') != '6',
    f.col('AA1[3]') != '7'
    ]
    weather_fltr = weather_post_split
    for i in fltr_msk:
      weather_fltr = weather_fltr.filter(i)

    #Reduce weather dataset to columns of interest (low level)
    weather_fltr_drop = weather_fltr.select('STATION','DATE','SOURCE','LATITUDE','LONGITUDE','WND[0]', 'WND[3]','VIS[0]','SLP[0]','AA1[0]')
    weather_fltr_drop = weather_fltr_drop.withColumnRenamed("DATE", "TIMESTAMP")

    return weather_fltr_drop

def distinct_station_fn(weather_fltr_drop):
    """
    For df input, return distinct stations for calculating closest stations to airports
    """
    weather_fltr_drop_distinct = weather_fltr_drop.select("STATION", "LATITUDE", "LONGITUDE").distinct()
    return weather_fltr_drop_distinct

def haversine_join_station_aircode_fn(airport_codes_df, weather_df):
    """
    For df input, return haversine distance
    """
    airport_codes_df.createOrReplaceTempView('airport_codes_us')
    weather_df.createOrReplaceTempView('stations_all')
    distance_query = "(SELECT airport_codes_us.code, stations_all.STATION, airport_codes_us.lat AS airport_lat, airport_codes_us.lon AS airport_lon, ( 3959 * acos(cos(radians(airport_codes_us.lat) ) * cos( radians( stations_all.LATITUDE ) ) * cos( radians( stations_all.LONGITUDE ) - radians(airport_codes_us.lon) ) + sin(radians(airport_codes_us.lat) ) * sin( radians( stations_all.LATITUDE ) ) ) ) AS airport_station_distance FROM airport_codes_us CROSS JOIN stations_all)"
    airports_stations_distance_all = spark.sql(distance_query)
    return airports_stations_distance_all
  
def airports_closest_stations_fn(airports_stations_distance_all):
    """
    For df input, return df with closest weather stations to airports
    """

    airports_stations_distance_all.createOrReplaceTempView('airports_stations_distance')
    closest_query = "(SELECT code AS airport_code, STATION AS station_name, airport_lat, airport_lon, airport_station_distance FROM airports_stations_distance ORDER BY airport_station_distance)"
    airports_closest_stations = spark.sql(closest_query)
  
    min_distance_query = "(SELECT code AS airport_code, STATION AS station_code, airport_lat, airport_lon, airport_station_distance FROM (SELECT *, row_number() over (partition by code order by airport_station_distance ASC) as seqnum from airports_stations_distance) airports_stations_distance where seqnum = 1)"
    airports_closest_station = spark.sql(min_distance_query)    

    MAX_ALLOWABLE_WEATHER_DISTANCE = 50.0
    airports_closest_station_filtered = airports_closest_station.filter(airports_closest_station.airport_station_distance < MAX_ALLOWABLE_WEATHER_DISTANCE)
    return airports_closest_station_filtered
  
def bearingClass_fn(flight_bearing, denominations=8):
    denom = 360/denominations
        
    if (int(flight_bearing) < 0 + denom/2) or (int(flight_bearing) > (7*denom) + (denom/2)):
      flight_bearing_class = "N"
    elif int(flight_bearing) <= denom + (denom/2):
      flight_bearing_class = "NW"
    elif int(flight_bearing) <= (2*denom) + (denom/2):
      flight_bearing_class = "W"
    elif int(flight_bearing) <= (3*denom) + (denom/2):
      flight_bearing_class = "SW"
    elif int(flight_bearing) <= (4*denom) + (denom/2):
      flight_bearing_class = "S"
    elif int(flight_bearing) <= (5*denom) + (denom/2):
      flight_bearing_class = "SE"
    elif int(flight_bearing) <= (6*denom) + (denom/2):
      flight_bearing_class = "E"
    elif int(flight_bearing) <= (7*denom) + (denom/2):
      flight_bearing_class = "NE"
    else:
      flight_bearing_class = "UNK"
      
    return flight_bearing_class
  
udfBearingClass_fn = udf(bearingClass_fn, StringType())

def bearingCalculation_fn(lat_a, lon_a, lat_b, lon_b):  
    lat_a_r, lat_b_r, lon_a_r, lon_b_r = radians(lat_a), radians(lat_b), radians(lon_a), radians(lon_b)
    delta_lon = lon_b - lon_a
    delta_lon_r = lon_b_r - lon_a_r
    X = cos(lat_b_r) * sin(delta_lon_r)
    Y = cos(lat_a_r) * sin(lat_b_r) - sin(lat_a_r) * cos(lat_b_r) * cos(delta_lon_r)
  
    flight_bearing = degrees(atan2(X, Y))
        
    flight_bearing_class = bearingClass_fn(flight_bearing)
  
    return flight_bearing_class
udfBearingCalculation_fn = udf(bearingCalculation_fn, StringType())

def join_closest_weather_airlines_fn(airlines_df, airports_closest_station_filtered):

    # add closest weather station to airlines dataset
    airlines_station_origin_filtered = airlines_df.join(airports_closest_station_filtered, airlines_df.ORIGIN == airports_closest_station_filtered.airport_code, how="inner")
    airlines_station_origin_filtered = airlines_station_origin_filtered.withColumnRenamed("station_code", "ORIGIN_STATION")
    airlines_station_origin_filtered = airlines_station_origin_filtered.withColumnRenamed("airport_station_distance", "ORIGIN_STATION_DISTANCE")
    airlines_station_origin_filtered = airlines_station_origin_filtered.withColumnRenamed("airport_lat", "ORIGIN_LAT")
    airlines_station_origin_filtered = airlines_station_origin_filtered.withColumnRenamed("airport_lon", "ORIGIN_LON")
    airlines_station_origin_filtered = airlines_station_origin_filtered.drop("airport_code")

    # add closest weather station to airlines dataset
    airlines_station_filtered = airlines_station_origin_filtered.join(airports_closest_station_filtered, airlines_station_origin_filtered.DEST == airports_closest_station_filtered.airport_code, how="inner")
    airlines_station_filtered = airlines_station_filtered.withColumnRenamed("station_code", "DEST_STATION")
    airlines_station_filtered = airlines_station_filtered.withColumnRenamed("airport_station_distance", "DEST_STATION_DISTANCE")
    airlines_station_filtered = airlines_station_filtered.withColumnRenamed("airport_lat", "DEST_LAT")
    airlines_station_filtered = airlines_station_filtered.withColumnRenamed("airport_lon", "DEST_LON")
    airlines_station_filtered = airlines_station_filtered.drop("airport_code")

    #add flight bearing angle in degrees from true north (consistent with wind direction)
    airlines_station_filtered = airlines_station_filtered.withColumn("FLIGHT_BEARING", udfBearingCalculation_fn("ORIGIN_LAT","ORIGIN_LON","DEST_LAT","DEST_LON"))
    return airlines_station_filtered

def flightDateTimeCalculation_fn(flight_date, flight_time):  
    timestamp_date = str(flight_date)
    timestamp_hour = str(flight_time).zfill(4)[:-2]
    timestamp_minute = str(flight_time).zfill(4)[-2:]
  
    timestamp = timestamp_date + 'T' + timestamp_hour + ':' + timestamp_minute# + ".000+0000"
    try:
      datetime_timestamp = dateutil.parser.isoparse(timestamp)
    except ValueError:
      timestamp = timestamp_date + 'T' + '00' + ':' + timestamp_minute# + ".000+0000"
      datetime_timestamp = dateutil.parser.isoparse(timestamp)
    
    return datetime_timestamp
  
def flightDateTimeCalculationArr_fn(flight_date, flight_time_dep, flight_time_arr):  
    timestamp_dep_date = str(flight_date)
    timestamp_arr_date = str(flight_date)
  
    
    timestamp_dep_hour = str(flight_time_dep).zfill(4)[:-2]
    timestamp_dep_minute = str(flight_time_dep).zfill(4)[-2:]
    timestamp_arr_hour = str(flight_time_arr).zfill(4)[:-2]
    timestamp_arr_minute = str(flight_time_arr).zfill(4)[-2:]
    
    timestamp_dep = timestamp_dep_hour + ':' + timestamp_dep_minute
    timestamp_arr = timestamp_arr_hour + ':' + timestamp_arr_minute
    
    timestamp_dep = timestamp_dep_date + 'T' + timestamp_dep_hour + ':' + timestamp_dep_minute# + ".000+0000"
    try:
      datetime_timestamp_dep = dateutil.parser.isoparse(timestamp_dep)
    except ValueError:
      timestamp_dep = timestamp_dep_date + 'T' + '00' + ':' + timestamp_dep_minute# + ".000+0000"
      datetime_timestamp_dep = dateutil.parser.isoparse(timestamp_dep)
    
    timestamp_arr = timestamp_arr_date + 'T' + timestamp_arr_hour + ':' + timestamp_arr_minute# + ".000+0000"
    try:
      datetime_timestamp_arr = dateutil.parser.isoparse(timestamp_arr)
    except ValueError:
      timestamp_arr = timestamp_arr_date + 'T' + '00' + ':' + timestamp_arr_minute# + ".000+0000"
      datetime_timestamp_arr = dateutil.parser.isoparse(timestamp_arr)
  
    # if flight arrived a later than when started, only works if flight was less than 24 hours long:
    if datetime_timestamp_dep > datetime_timestamp_arr:
      datetime_timestamp_arr = datetime_timestamp_arr + datetime.timedelta(days=1)

    return datetime_timestamp_arr

udfFlightDateTimeCalculation_fn = udf(flightDateTimeCalculation_fn, TimestampType())
udfFlightDateTimeCalculationArr_fn = udf(flightDateTimeCalculationArr_fn, TimestampType())

def airlines_station_datetime_fn(airlines_station_filtered):
    airlines_station_datetime = airlines_station_filtered.withColumn("CRS_DEP_TIMESTAMP", udfFlightDateTimeCalculation_fn("FL_DATE","CRS_DEP_TIME"))
    airlines_station_datetime = airlines_station_datetime.withColumn("CRS_ARR_TIMESTAMP", udfFlightDateTimeCalculationArr_fn("FL_DATE","CRS_DEP_TIME", "CRS_ARR_TIME"))
    return airlines_station_datetime

def airlines_station_datetime_unix_fn(airlines_station_datetime):
    airlines_station_datetime_unix = airlines_station_datetime.withColumn("CRS_DEP_TIMESTAMP_UNIX", f.unix_timestamp("CRS_DEP_TIMESTAMP"))
    airlines_station_datetime_unix = airlines_station_datetime_unix.withColumn("CRS_ARR_TIMESTAMP_UNIX", f.unix_timestamp("CRS_ARR_TIMESTAMP"))
    airlines_station_datetime_unix = airlines_station_datetime_unix.withColumn("DEP_HOUR", f.hour("CRS_DEP_TIMESTAMP"))
    airlines_station_datetime_unix = airlines_station_datetime_unix.withColumn("ARR_HOUR", f.hour("CRS_ARR_TIMESTAMP"))
    
    return airlines_station_datetime_unix
  
def weather_fltr_datetime_fn(weather_fltr_drop):
    weather_fltr_datetime = weather_fltr_drop.withColumn("DATE_TIMESTAMP_UNIX", f.unix_timestamp("TIMESTAMP"))
    weather_fltr_datetime = weather_fltr_datetime.withColumn('DATE', f.col("TIMESTAMP").cast(DateType()))
    weather_fltr_datetime = weather_fltr_datetime.withColumn("HOUR", f.hour("TIMESTAMP"))
    
    return weather_fltr_datetime

def weather_avg_fn(weather_fltr_datetime):
    weather_fltr_datetime.createOrReplaceTempView('weather_time')
    weather_avg_query = "(SELECT STATION, DATE, HOUR, ROUND(AVG(`WND[0]`),0) AS `WND[0]`, ROUND(AVG(`WND[3]`),0) AS `WND[3]`, ROUND(AVG(`VIS[0]`),0) AS `VIS[0]`, ROUND(AVG(`SLP[0]`),0) AS `SLP[0]`, ROUND(AVG(`AA1[0]`),0) AS `AA1[0]` FROM weather_time GROUP BY STATION, DATE, HOUR)"

    weather_avg = spark.sql(weather_avg_query)
    
    weather_avg = weather_avg.withColumn("WND_CLASS[0]", udfBearingClass_fn("WND[0]"))
    weather_avg = weather_avg.drop("WND[0]")
    
    return weather_avg
  
def weather_add_values_fn(weather_avg):
    weather_fltr_datetime_origin = weather_avg.withColumnRenamed("STATION", "ORIGIN_STATION_WEATHER")
    weather_fltr_datetime_origin = weather_fltr_datetime_origin.withColumnRenamed("DATE", "ORIGIN_STATION_DATE")
    weather_fltr_datetime_origin = weather_fltr_datetime_origin.withColumnRenamed("HOUR", "ORIGIN_STATION_HOUR")
    weather_fltr_datetime_origin = weather_fltr_datetime_origin.withColumnRenamed("WND_CLASS[0]", "ORIGIN_STATION_WND[0]")
    weather_fltr_datetime_origin = weather_fltr_datetime_origin.withColumnRenamed("WND[3]", "ORIGIN_STATION_WND[3]")
    weather_fltr_datetime_origin = weather_fltr_datetime_origin.withColumnRenamed("VIS[0]", "ORIGIN_STATION_VIS[0]")
    weather_fltr_datetime_origin = weather_fltr_datetime_origin.withColumnRenamed("SLP[0]", "ORIGIN_STATION_SLP[0]")
    weather_fltr_datetime_origin = weather_fltr_datetime_origin.withColumnRenamed("AA1[0]", "ORIGIN_STATION_AA1[0]")
    weather_fltr_datetime_dest = weather_avg.withColumnRenamed("STATION", "DEST_STATION_WEATHER")
    weather_fltr_datetime_dest = weather_fltr_datetime_dest.withColumnRenamed("DATE", "DEST_STATION_DATE")
    weather_fltr_datetime_dest = weather_fltr_datetime_dest.withColumnRenamed("HOUR", "DEST_STATION_HOUR")
    weather_fltr_datetime_dest = weather_fltr_datetime_dest.withColumnRenamed("WND_CLASS[0]", "DEST_STATION_WND[0]")
    weather_fltr_datetime_dest = weather_fltr_datetime_dest.withColumnRenamed("WND[3]", "DEST_STATION_WND[3]")
    weather_fltr_datetime_dest = weather_fltr_datetime_dest.withColumnRenamed("VIS[0]", "DEST_STATION_VIS[0]")
    weather_fltr_datetime_dest = weather_fltr_datetime_dest.withColumnRenamed("SLP[0]", "DEST_STATION_SLP[0]")
    weather_fltr_datetime_dest = weather_fltr_datetime_dest.withColumnRenamed("AA1[0]", "DEST_STATION_AA1[0]")
    return weather_fltr_datetime_origin, weather_fltr_datetime_dest
  
def departure_final_fn(airlines_station_datetime_unix):
    airlines_station_datetime_unix.createOrReplaceTempView("airports_weather")
    weather_fltr_datetime_origin.createOrReplaceTempView("origin_weather")
    origin_join_query = "(SELECT * FROM airports_weather a INNER JOIN origin_weather w ON a.ORIGIN_STATION = w.ORIGIN_STATION_WEATHER AND a.FL_DATE = w.ORIGIN_STATION_DATE AND a.DEP_HOUR = w.ORIGIN_STATION_HOUR)"

    departure_final = spark.sql(origin_join_query)
    return departure_final

def airlines_weather_final_trim_fn(departure_final):
    departure_final.createOrReplaceTempView("airports_weather_dest")
    weather_fltr_datetime_dest.createOrReplaceTempView("dest_weather")
    # chnaged to join on weather @ destination airport @ departure time
    dest_join_query = "(SELECT * FROM airports_weather_dest a INNER JOIN dest_weather w ON a.DEST_STATION = w.DEST_STATION_WEATHER AND a.FL_DATE = w.DEST_STATION_DATE AND a.DEP_HOUR = w.DEST_STATION_HOUR)"

    airlines_weather_final = spark.sql(dest_join_query)
    drop_cols = ['DEST_STATION_DATE', 'DEST_STATION_HOUR', 'ORIGIN_STATION_HOUR', 'ORIGIN_STATION_DATE', 'ORIGIN_LAT', 'ORIGIN_LON', 'DEST_LAT', 'DEST_LON', 'CRS_DEP_TIMESTAMP_UNIX', 'CRS_ARR_TIMESTAMP_UNIX', 'DEP_HOUR', 'ARR_HOUR', 'ORIGIN_STATION', 'DEST_STATION', 'ORIGIN_STATION_WEATHER', 'DEST_STATION_WEATHER']
    airlines_weather_final_trim = airlines_weather_final.drop(*drop_cols)
    return airlines_weather_final_trim
  
def airlines_weather_to_parquet_fn(airlines_weather_final_trim):
    dbutils.fs.rm("dbfs:/tmp/parquet/airlines_weather_final_4_7.parquet")
    airlines_weather_final_trim.write.parquet("dbfs:/tmp/parquet/airlines_weather_final_4_7.parquet")
    return None

# COMMAND ----------

airlines_df =  preprocessAirlines(airlines)
weather_parquet_us = US_fn(weather_parquet)
weather_fltr_drop = reduce_split_cols_fn(weather_parquet_us)
weather_fltr_drop_distinct = distinct_station_fn(weather_fltr_drop)
airport_codes_us = US_fn(airport_codes)
airports_stations_distance_all = haversine_join_station_aircode_fn(airport_codes_us, weather_fltr_drop_distinct)
airports_closest_station_filtered = airports_closest_stations_fn(airports_stations_distance_all)
airports_closest_station_filtered = join_closest_weather_airlines_fn(airlines_df, airports_closest_station_filtered)
airlines_station_datetime = airlines_station_datetime_fn(airports_closest_station_filtered)
airlines_station_datetime_unix = airlines_station_datetime_unix_fn(airlines_station_datetime)  
weather_fltr_datetime = weather_fltr_datetime_fn(weather_fltr_drop)
weather_avg = weather_avg_fn(weather_fltr_datetime)
weather_fltr_datetime_origin, weather_fltr_datetime_dest = weather_add_values_fn(weather_avg)
departure_final = departure_final_fn(airlines_station_datetime_unix)
airlines_weather_final_trim = airlines_weather_final_trim_fn(departure_final)
display(airlines_weather_final_trim)

# COMMAND ----------

airlines_weather_final_trim.write.parquet("/FileStore/tables/airlines_weather_final_trim.parquet")
airlines_weather_final_trim = spark.read.parquet("/FileStore/tables/airlines_weather_final_trim.parquet")

# COMMAND ----------

airlines_weather_final_trim.printSchema()

# COMMAND ----------

# MAGIC %md ## Data Split:

# COMMAND ----------

airlines_train, airlines_val, airlines_test = airlines_weather_final_trim.randomSplit([0.8,0.1,0.1], seed = 2020)

# COMMAND ----------

# train_cnt = airlines_train.count()
# val_cnt = airlines_val.count()
# test_cnt = airlines_test.count()
# total_cnt = train_cnt + val_cnt + test_cnt
# print('airlines_train records: {}\n airlines_val records: {}\n  airlines_test records: {}\n total records: {}'.format(train_cnt, val_cnt, test_cnt, total_cnt) )

# COMMAND ----------

# MAGIC %md ## Feature Engineering & Feature Selection:

# COMMAND ----------

def featureSelection(df):
  cols_to_keep = ['MONTH', 'DAY_OF_WEEK', 'DEP_DELAY', 'DEP_TIME_BLK', 'ARR_DELAY', 'ARR_TIME_BLK', 'CRS_ELAPSED_TIME', 'DISTANCE',  'CARRIER_DELAY', 'WEATHER_DELAY', 
   'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'IS_WEEKEND', 'DEP_RUSH_HOUR', 'ARR_RUSH_HOUR', 'DEP_TIME', 'CRS_DEP_TIME', 'ARR_TIME', 'CRS_ARR_TIME', 'FLIGHT_BEARING', 'ORIGIN_STATION_WND_DIR', 'ORIGIN_STATION_VIS', 'ORIGIN_STATION_SLP','ORIGIN_STATION_AA1',    'ORIGIN_STATION_WND', 'DEST_STATION_WND_DIR', 'DEST_STATION_VIS', 'DEST_STATION_SLP', 'DEST_STATION_AA1',  'DEST_STATION_WND']
  cols_to_remove = [x for x in df.columns if x not in cols_to_keep]
  df = df.withColumnRenamed("ORIGIN_STATION_WND[0]", "ORIGIN_STATION_WND_DIR")
  df = df.withColumnRenamed("ORIGIN_STATION_VIS[0]", "ORIGIN_STATION_VIS")
  df = df.withColumnRenamed("ORIGIN_STATION_SLP[0]", "ORIGIN_STATION_SLP")
  df = df.withColumnRenamed("ORIGIN_STATION_AA1[0]", "ORIGIN_STATION_AA1")
  df = df.withColumnRenamed("ORIGIN_STATION_WND[3]", "ORIGIN_STATION_WND")
  
  df = df.withColumnRenamed("DEST_STATION_WND[0]", "DEST_STATION_WND_DIR")
  df = df.withColumnRenamed("DEST_STATION_VIS[0]", "DEST_STATION_VIS")
  df = df.withColumnRenamed("DEST_STATION_SLP[0]", "DEST_STATION_SLP")
  df = df.withColumnRenamed("DEST_STATION_AA1[0]", "DEST_STATION_AA1")
  df = df.withColumnRenamed("DEST_STATION_WND[3]", "DEST_STATION_WND")  
  
  featureSelection_df = df.drop(*cols_to_remove)
  return featureSelection_df

# COMMAND ----------

airlines_train_df =  featureSelection(airlines_train)
airlines_train_df.printSchema()

# COMMAND ----------

sampleDF = airlines_train_df.sample(False, 0.0000001)
display(sampleDF)

# COMMAND ----------

sampleDF

# COMMAND ----------

# def nullDataFrame(df):
#   null_feature_list = []
#   count = df.count()
#   for column in df.columns:
#     nulls = df.filter(df[column].isNull()).count()
#     nulls_perct = np.round((nulls/count)*100, 2)
#     null_feature_list.append([column, nulls, nulls_perct])
#   nullCounts_df = pd.DataFrame(np.array(null_feature_list), columns=['Feature_Name', 'Null_Counts', 'Percentage_Null_Counts'])
#   return nullCounts_df


# nullCounts_airlines_train_df = nullDataFrame(airlines_train_df)
# nullCounts_airlines_train_df

# COMMAND ----------

numeric_features = [x[0] for x in airlines_train_df.dtypes if x[1] == 'int' or x[1] == 'double']
numeric_features.remove('ARR_DELAY')
numeric_features

# COMMAND ----------

categorical_features = [x[0] for x in airlines_train_df.dtypes if x[1] == 'string']
categorical_features

# COMMAND ----------

stages = []
for categoricalCol in categorical_features:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index', handleInvalid="keep")
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
assemblerInputs = [c + "classVec" for c in categorical_features] + numeric_features
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features", handleInvalid="keep")
stages += [assembler]

# COMMAND ----------

pipeline = Pipeline().setStages(stages).fit(airlines_train_df)
vector_airlines_train_df = pipeline.transform(airlines_train_df)
vector_airlines_train_df.printSchema()

# COMMAND ----------

# MAGIC %md ## Models:
# MAGIC To predict `ARRIVAL_DELAY` from the dataset, we are going to consider below supervised machine learning algorithms using cross validation.
# MAGIC 
# MAGIC 1. Linear Regression
# MAGIC 2. Decision Tree Regressor
# MAGIC 3. Random Forest Regressor
# MAGIC 4. Gradient Boosted Tree Regressor

# COMMAND ----------

# MAGIC %md ### Linear Regression:

# COMMAND ----------

train_df = vector_airlines_train_df.select(col("ARR_DELAY").alias("label"), col("features"))
train_df.show(2)

# COMMAND ----------

airlines_val_df =  featureSelection(airlines_val)

# COMMAND ----------

vector_airlines_val_df = pipeline.transform(airlines_val_df)
val_df = vector_airlines_val_df.select(col("ARR_DELAY").alias("label"), col("features"))

# COMMAND ----------

lr = LinearRegression(featuresCol = 'features', labelCol='label')
paramGrid_lr = ParamGridBuilder() \
   .addGrid(lr.regParam, [0.1, 0.01, 0.001]) \
   .addGrid(lr.maxIter, [10, 20]) \
   .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
   .build() 

crossval_lr = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid_lr,
                          evaluator=RegressionEvaluator(),
                          numFolds=5) 

cvModel_lr = crossval_lr.fit(train_df)

# COMMAND ----------

regression_evaluator_r2 = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="r2")
regression_evaluator_rmse = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="rmse")
regression_evaluator_mae = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="mae")

regression_metrics_list = []

# COMMAND ----------

# Save and Load CrossValidatorModel
cvModel_lr.write().overwrite().save('/FileStore/tables/cvModel_lr')

# COMMAND ----------

saved_cvModel_lr = CrossValidatorModel.load('/FileStore/tables/cvModel_lr')

# COMMAND ----------

# train_df evaluation metrics
lr_predictions_train = saved_cvModel_lr.transform(train_df)
lr_train_r2 = regression_evaluator_r2.evaluate(lr_predictions_train)
lr_train_rmse = regression_evaluator_rmse.evaluate(lr_predictions_train)
lr_train_mae = regression_evaluator_mae.evaluate(lr_predictions_train)
regression_metrics_list.append(["LinearRegression_TrainData_CV", lr_train_r2, lr_train_rmse, lr_train_mae ])

# COMMAND ----------

# # val_df evaluation metrics
lr_predictions_val = saved_cvModel_lr.transform(val_df)
lr_val_r2 = regression_evaluator_r2.evaluate(lr_predictions_val)
lr_val_rmse = regression_evaluator_rmse.evaluate(lr_predictions_val)
lr_val_mae = regression_evaluator_mae.evaluate(lr_predictions_val)
regression_metrics_list.append(["LinearRegression_ValData_CV", lr_val_r2, lr_val_rmse, lr_val_mae ])

# COMMAND ----------

regression_metrics_df = pd.DataFrame(regression_metrics_list, columns = ['Model_Data' , 'R^2', 'RMSE', 'MAE']) 
display(regression_metrics_df)

# COMMAND ----------

bestLRModel = cvModel_lr.bestModel
bestParams = bestLRModel.extractParamMap()
bestParams

# COMMAND ----------

# MAGIC %md ### Decision Tree Regressor

# COMMAND ----------

dt = DecisionTreeRegressor(featuresCol="features", labelCol='label') 

paramGrid_dt = ParamGridBuilder()\
    .addGrid(dt.maxBins, [16, 32]) \
    .addGrid(dt.maxDepth, [5, 10]) \
    .addGrid(dt.minInstancesPerNode, [1, 5]) \
    .build()  

crossval_dt = CrossValidator(estimator=dt,
                          estimatorParamMaps=paramGrid_dt,
                          evaluator=RegressionEvaluator(),
                          numFolds=5) 

cvModel_dt = crossval_dt.fit(train_df)

# COMMAND ----------

# Save and Load CrossValidatorModel
cvModel_dt.write().overwrite().save('/FileStore/tables/cvModel_dt')

# COMMAND ----------

saved_cvModel_dt = CrossValidatorModel.load('/FileStore/tables/cvModel_dt')

# COMMAND ----------

# train_df evaluation metrics
dt_predictions_train = saved_cvModel_dt.transform(train_df)
dt_train_r2 = regression_evaluator_r2.evaluate(dt_predictions_train)
dt_train_rmse = regression_evaluator_rmse.evaluate(dt_predictions_train)
dt_train_mae = regression_evaluator_mae.evaluate(dt_predictions_train)
regression_metrics_list.append(["DecisionTreeRegressor_TrainData_CV", dt_train_r2, dt_train_rmse, dt_train_mae ])

# COMMAND ----------

# val_df evaluation metrics
dt_predictions_val = saved_cvModel_dt.transform(val_df)
dt_val_r2 = regression_evaluator_r2.evaluate(dt_predictions_val)
dt_val_rmse = regression_evaluator_rmse.evaluate(dt_predictions_val)
dt_val_mae = regression_evaluator_mae.evaluate(dt_predictions_val)
regression_metrics_list.append(["DecisionTreeRegressor_ValData_CV", dt_val_r2, dt_val_rmse, dt_val_mae ])

# COMMAND ----------

bestDTModel = cvModel_dt.bestModel
bestParams_dt = bestDTModel.extractParamMap()
bestParams_dt

# COMMAND ----------

# MAGIC %md ### Random Forest Regressor

# COMMAND ----------

rf = RandomForestRegressor(featuresCol="features", labelCol='label')

paramGrid_rf = ParamGridBuilder()\
    .addGrid(rf.maxBins, [16, 32]) \
    .addGrid(rf.numTrees, [20, 40]) \
    .addGrid(rf.minInstancesPerNode, [1, 5]) \
    .build()  

crossval_rf = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid_rf,
                          evaluator=RegressionEvaluator(),
                          numFolds=5) 

cvModel_rf = crossval_rf.fit(train_df)

# COMMAND ----------

# Save and Load CrossValidatorModel
cvModel_rf.write().overwrite().save('/FileStore/tables/cvModel_rf')


# COMMAND ----------

saved_cvModel_rf = CrossValidatorModel.load('/FileStore/tables/cvModel_rf')

# COMMAND ----------

rf_predictions_train = saved_cvModel_rf.transform(train_df)
rf_train_r2 = regression_evaluator_r2.evaluate(rf_predictions_train)
rf_train_rmse = regression_evaluator_rmse.evaluate(rf_predictions_train)
rf_train_mae = regression_evaluator_mae.evaluate(rf_predictions_train)
regression_metrics_list.append(["RandomForestRegressor_TrainData_CV", rf_train_r2, rf_train_rmse, rf_train_mae ])

# COMMAND ----------

# val_df evaluation metrics
rf_predictions_val = saved_cvModel_rf.transform(val_df)
rf_val_r2 = regression_evaluator_r2.evaluate(rf_predictions_val)
rf_val_rmse = regression_evaluator_rmse.evaluate(rf_predictions_val)
rf_val_mae = regression_evaluator_mae.evaluate(rf_predictions_val)
regression_metrics_list.append(["RandomForestRegressor_ValData_CV", rf_val_r2, rf_val_rmse, rf_val_mae ])

# COMMAND ----------

bestRFModel = cvModel_rf.bestModel
bestParams_rf = bestRFModel.extractParamMap()
bestParams_rf

# COMMAND ----------

# MAGIC %md ### Gradient-Boosted Trees

# COMMAND ----------

 gbt = GBTRegressor(featuresCol="features", labelCol='label')

paramGrid_gbt = ParamGridBuilder()\
    .addGrid(gbt.maxBins, [10, 32]) \
    .addGrid(gbt.minInstancesPerNode, [1, 5]) \
    .build()  

crossval_gbt = CrossValidator(estimator=gbt,
                          estimatorParamMaps=paramGrid_gbt,
                          evaluator=RegressionEvaluator(),
                          numFolds=5) 

cvModel_gbt = crossval_gbt.fit(train_df)

# COMMAND ----------

# Save and Load CrossValidatorModel
cvModel_gbt.write().overwrite().save('/FileStore/tables/cvModel_gbt')


# COMMAND ----------

saved_cvModel_gbt = CrossValidatorModel.load('/FileStore/tables/cvModel_gbt')

# COMMAND ----------

gbt_predictions_train = saved_cvModel_gbt.transform(train_df)
gbt_train_r2 = regression_evaluator_r2.evaluate(gbt_predictions_train)
gbt_train_rmse = regression_evaluator_rmse.evaluate(gbt_predictions_train)
gbt_train_mae = regression_evaluator_mae.evaluate(gbt_predictions_train)
regression_metrics_list.append(["GradientBoostedTreeRegressor_TrainData_CV", gbt_train_r2, gbt_train_rmse, gbt_train_mae ])

# COMMAND ----------

# val_df evaluation metrics
gbt_predictions_val = saved_cvModel_gbt.transform(val_df)
gbt_val_r2 = regression_evaluator_r2.evaluate(gbt_predictions_val)
gbt_val_rmse = regression_evaluator_rmse.evaluate(gbt_predictions_val)
gbt_val_mae = regression_evaluator_mae.evaluate(gbt_predictions_val)
regression_metrics_list.append(["GradientBoostedTreeRegressor_ValData_CV", gbt_val_r2, gbt_val_rmse, gbt_val_mae ])

# COMMAND ----------

bestGBTModel = cvModel_gbt.bestModel
bestParams_gbt = bestGBTModel.extractParamMap()
bestParams_gbt

# COMMAND ----------

# MAGIC %md ## Results: 

# COMMAND ----------

regression_metrics_df = pd.DataFrame(regression_metrics_list, columns = ['Model_Data' , 'R^2', 'RMSE', 'MAE']) 
display(regression_metrics_df)

# COMMAND ----------

# MAGIC %md ## Inference on Test Data:

# COMMAND ----------

airlines_test_df =  featureSelection(airlines_test)

# COMMAND ----------

vector_airlines_test_df = pipeline.transform(airlines_test_df)
test_df = vector_airlines_test_df.select(col("ARR_DELAY").alias("label"), col("features"))

# COMMAND ----------

# # test_df evaluation metrics
lr_predictions_test = saved_cvModel_lr.transform(test_df)
lr_test_r2 = regression_evaluator_r2.evaluate(lr_predictions_test)
lr_test_rmse = regression_evaluator_rmse.evaluate(lr_predictions_test)
lr_test_mae = regression_evaluator_mae.evaluate(lr_predictions_test)
regression_metrics_list.append(["LinearRegression_TestData_CV", lr_test_r2, lr_test_rmse, lr_test_mae ])

# COMMAND ----------

regression_metrics_df = pd.DataFrame(regression_metrics_list, columns = ['Model_Data' , 'R^2', 'RMSE', 'MAE']) 
display(regression_metrics_df)

# COMMAND ----------

# MAGIC %md ## Conclusion:

# COMMAND ----------

