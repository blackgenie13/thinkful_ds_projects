# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 23:52:07 2015

@author: Michael Lin_2
"""

######  EXTRACTING THE DATA FROM THE WEB  ######

import requests
import collections
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import pandas as pd

## Grabbing the JSON (JavaScript Object Notation) file directly from the web
r = requests.get('http://www.citibikenyc.com/stations/json')

r.text            # view the json file in text, which is just a long string
r.json()          # view the json file in JSON format - same as dictionary
r.json().keys()   # view all the key in the strcture extracted

## This loop will test to see if we have all the fiels from 'stationBeanList'
## and it will copy all the 'stationBeanList' columns into the list key_list[]
# NOTE: theoratically, you do not need the rist for loop because all rows will
# have the same number of column.... this is just to ensure in case not all
# rows have the same columns

key_list = []       #unique list of keys for each station listing
for station in r.json()['stationBeanList']:
    for k in station.keys():
        if k not in key_list:
            key_list.append(k)

# Alternatively, we can achieve the same here
key_list2 = []
for k in r.json()['stationBeanList'][0]:
        if k not in key_list2:
            key_list2.append(k)

# Remember r.json()['stationBeanList'] is a list
r.json()['stationBeanList'][0]

## Check the range of 'availableBikes' and 'totalDocks'
df = json_normalize(r.json()['stationBeanList'])
df['availableBikes'].hist()
plt.show()
df['totalDocks'].hist()
plt.show()

## Compute the mean and median of 'totalDocks'
df['totalDocks'].mean()
df['totalDocks'].median()

## Compute the mean and median of 'totalDocks' that are in service
df[df['statusValue'] == 'In Service']['totalDocks'].mean()
df[df['statusValue'] == 'In Service']['totalDocks'].median()
# Alternative method
condition = (df['statusValue'] == 'In Service')
totalDocks_mean = df[condition]['totalDocks'].mean()
totalDocks_median = df[condition]['totalDocks'].median()
print ('The average # of total docks in service is ' + str(totalDocks_mean))
print ('The median # of total docks in service is ' + str(totalDocks_median))

## Test to see if there are any 'test' station
# len(r.json()['stationBeanList'])        # Total records
collections.Counter(df['testStation']==True)  # Number of non-test records

## Compute the average and median of 'available bikes'
print ('The average # of available bikes is ' + str(df['availableBikes'].mean()))
print ('The median # of available bikes is ' + str(df['availableBikes'].median()))

## Compute the average and median of 'available bikes' that are in service
condition2 = (df['statusValue'] == 'In Service')
print ('The mean # of available bikes from in-service docks is ' + str(df[condition2]['availableBikes'].mean()))
print ('The median # of available bikes from in-service docks is ' + str(df[condition2]['availableBikes'].median()))


######  STORE THE DATA INTO A DATABASE  ######

import sqlite3 as lite

## Setup or connect to the database, note the db will be saved in the directory location if not already
con = lite.connect('citibike_test.db')
cur = con.cursor()

## Create the table with attributes
with con:
    cur.execute('CREATE TABLE citibike_reference \
    (id INT PRIMARY KEY, totalDocks INT, city TEXT, altitude INT, \
    stAddress2 TEXT, longitude NUMERIC, postalCode TEXT, \
    testStation TEXT, stAddress1 TEXT,  stationName TEXT, \
    landMark TEXT, latitude NUMERIC, location TEXT)')

# A prepared SQL statement we're going to execute over and over again
sql = "INSERT INTO citibike_reference \
    (id, totalDocks, city, altitude, stAddress2, longitude, \
    postalCode, testStation, stAddress1, stationName, landMark, \
    latitude, location) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"

# This loop is to populate values in the database
with con:
    for station in r.json()['stationBeanList']:
        # id, totalDocks, city, altitude, stAddress2, longitude, postalCode, 
        # testStation, stAddress1, stationName, landMark, latitude, location)
        cur.execute(sql,\
        (station['id'],station['totalDocks'],station['city'],\
        station['altitude'],station['stAddress2'],station['longitude'],\
        station['postalCode'],station['testStation'],station['stAddress1'],\
        station['stationName'],station['landMark'],station['latitude'],station['location']))

# Extract the id from 'id' column of the DataFrame and put them into a list
station_ids = df['id'].tolist() 

# Add the '_' to the station name and also add the data type for SQLite
# Note the we cannot start the column name with number, hence the '_'
station_ids = ['_' + str(x) + ' INT' for x in station_ids]

# Create the table - in this case, we're concatentating the string and joining
# all the station ids (now with '_' and 'INT' added)
with con:
    cur.execute("CREATE TABLE available_bikes ( execution_time INT, " +  ", ".join(station_ids) + ");")


import time                         # Import a package with datetime objects
import datetime                     # Import this package for converting datetime string format to int
from dateutil.parser import parse   # A package for parsing a string into a Python datetime object

# Take the string and parse it into a Python datetime object
# Note that this is the script Execution Time
exec_time = parse(r.json()['executionTime'])

## Insert the very first script execution time, but formatted in total seceonds since 1/1/1970
with con:
    cur.execute('INSERT INTO available_bikes (execution_time) VALUES (?)', \
    (str((exec_time - datetime.datetime(1970,1,1)).total_seconds()),))
    
# Defaultdict to store available bikes by station
id_bikes = collections.defaultdict(int)

# Loop through the stations in the station list to store the available bikes value
for station in r.json()['stationBeanList']:
    id_bikes[station['id']] = station['availableBikes']

# Iterate through the defaultdict to update the values in the database
with con:
    for k, v in id_bikes.items():
        cur.execute("UPDATE available_bikes SET _" + str(k) + " = " + str(v) + \
        " WHERE execution_time = " + str((exec_time - datetime.datetime(1970,1,1)).total_seconds()) + ";")


######  SET THIS UP SO IT RECORD THE DATA EVERY MINUTE FOR 60 MINUTES  ######

con = lite.connect('citibike.db')
cur = con.cursor()

for i in range(60):
    r = requests.get('http://www.citibikenyc.com/stations/json')
    exec_time = parse(r.json()['executionTime'])

    cur.execute('INSERT INTO available_bikes (execution_time) VALUES (?)', (str((exec_time - datetime.datetime(1970,1,1)).total_seconds()),))
    con.commit()

    id_bikes = collections.defaultdict(int)
    for station in r.json()['stationBeanList']:
        id_bikes[station['id']] = station['availableBikes']

    for k, v in id_bikes.items():
        cur.execute("UPDATE available_bikes SET _" + str(k) + " = " + str(v) + " WHERE execution_time = " + str((exec_time - datetime.datetime(1970,1,1)).total_seconds()) + ";")
    con.commit()

    time.sleep(60)

con.close() #close the database connection when done

"""
######  ALTERNATIVE LOOP  ######
for i in range(60):
    r = requests.get('http://www.citibikenyc.com/stations/json')
    exec_time = parse(r.json()['executionTime'])

    cur.execute('INSERT INTO available_bikes (execution_time) VALUES (?)', (str((exec_time - datetime.datetime(1970,1,1)).total_seconds()),))

    for station in r.json()['stationBeanList']:
        cur.execute("UPDATE available_bikes SET _%d = %d WHERE execution_time = %s" % (station['id'], station['availableBikes'], str((exec_time - datetime.datetime(1970,1,1)).total_seconds())))
    con.commit()

    time.sleep(60)

con.close() #close the database connection when done
"""

con = lite.connect('citibike_test.db')
cur = con.cursor()

## Create the dataframe based on the table available_bikes from the connected DB
df = pd.read_sql_query("SELECT * FROM available_bikes ORDER BY execution_time",con,index_col='execution_time')

## CALCULATE THE TOTAL CHNAGES WITHIN THE HOUR  FOR EACH STATION
hour_change = collections.defaultdict(int)   # create a dictionary for hour change
for col in df.columns:                       # for col in each column/station id
    station_vals = df[col].tolist()          # write all the available bike values of current station to a list
    station_id = col[1:]                     # trim the "_" of the current station id
    station_change = 0                       # initiate a variable
    for k,v in enumerate(station_vals):      # for k in range(len(station_vals))
        if k < len(station_vals) - 1:        # only do this until the row immediately prior to the last row
            station_change += abs(station_vals[k] - station_vals[k+1])
    hour_change[int(station_id)] = station_change # convert the station id back to integer
    

def keywithmaxval(d):
    """Find the key with the greatest value"""
    return max(d, key=lambda k: d[k])

# assign the max key to max_station
max_station = keywithmaxval(hour_change)
print ('The station that has the highest change is Station ID #' + str(max_station) + ' with ' + str(hour_change[max_station]) + ' of chnages within the last hour')


# query sqlite for reference information
import datetime
cur.execute("SELECT id, stationname, latitude, longitude FROM citibike_reference WHERE id = ?", (max_station,))
data = cur.fetchone()     # Fetch the next row of a query result set, return a single tuple (list in a list)
print("The most active station is station id %s at %s latitude: %s longitude: %s " % data)
print("With %d bicycles coming and going in the hour between %s and %s" % (
    hour_change[max_station],
    datetime.datetime.fromtimestamp(int(df.index[0])).strftime('%Y-%m-%d T%H:%M:%S'),
    datetime.datetime.fromtimestamp(int(df.index[-1])).strftime('%Y-%m-%d T%H:%M:%S'),
))

## Visually Inspect the Data
import matplotlib.pyplot as plt
plt.bar(hour_change.keys(), hour_change.values())
plt.show()
plt.savefig('histogram_citibike.png')