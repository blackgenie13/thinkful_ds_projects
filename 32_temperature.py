# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 20:35:21 2015

@author: Michael Lin_2
"""

import requests
import sqlite3 as lite
import datetime

# Testing the API Extraction
# https://api.forecast.io/forecast/07d377d5071bedce9e004d890514e76a/37.727239,-123.032229,2015-11-22T12:00:00 

## Define the API URL with our access token - see: https://developer.forecast.io/
api_key = "07d377d5071bedce9e004d890514e76a/"
url = 'https://api.forecast.io/forecast/' + api_key

## Define the cities in a dictionary
cities = { "Atlanta": '33.762909,-84.422675',
            "Austin": '30.303936,-97.754355',
            "Boston": '42.331960,-71.020173',
            "Chicago": '41.837551,-87.681844',
            "San_Francisco": '37.727239,-123.032229' }

# By setting this equal to a variable, we fix the calculation to the point when 
# we started the scrip (rather than have things move aroudn while we're coding.)
end_date = datetime.datetime.now() 

## Constrct the Database
con = lite.connect('weather.db')
cur = con.cursor()

cities.keys()
# First we creat the table
with con:
    cur.execute('CREATE TABLE daily_temp \
    ( day_of_reading INT, \
    Atlanta REAL, Austin REAL, Boston REAL, Chicago REAL, San_Francisco REAL);') 
    
# Second we create a date variable for query purpose
query_date = end_date - datetime.timedelta(days=30) #the current value being processed

# Insert all the 30 days of query first - note that we're using 12:00:00 as our time instead of the end_date time
with con:
    while query_date < end_date:
        cur.execute("INSERT INTO daily_temp(day_of_reading) VALUES (?)", (str(query_date.strftime('%Y-%m-%dT12:00:00')),))
        query_date += datetime.timedelta(days=1)
        
# The loop will record each major city with 30 days of temperature.
# Note that cities.items() is a tuples
# Note that we're only extracting r.json()['daily']['data'][0]['temperatureMax'], which is the highest temperature of the day
for k,v in cities.items():
    query_date = end_date - datetime.timedelta(days=30) # Reset value each time through the loop of cities
    while query_date < end_date:
        # Request and query for the value
        r = requests.get(url + v + ',' +  query_date.strftime('%Y-%m-%dT12:00:00'))

        with con:
            # Insert the temperature max into the table based on the city name
            cur.execute('UPDATE daily_temp SET ' + k + ' = ' + \
            str(r.json()['daily']['data'][0]['temperatureMax']) + \
            " WHERE day_of_reading = (?)", (str(query_date.strftime('%Y-%m-%dT12:00:00')),))

        # Increment query_date to the next day for next operation of loop
        query_date += datetime.timedelta(days=1) #increment query_date to the next day

con.close() # a good practice to close connection to database



import pandas as pd
import collections

con = lite.connect('weather.db')
cur = con.cursor()

df = pd.read_sql_query("SELECT * FROM daily_temp ORDER BY day_of_reading",con,index_col='day_of_reading')

day_change = collections.defaultdict(str)
max_temp = collections.defaultdict(str)
min_temp = collections.defaultdict(str)
var_temp = collections.defaultdict(str)

for col in df.columns:
    city_vals = df[col].tolist()
    city_change = 0
    maxv = city_vals[0]
    minv = city_vals[0]
    for k, v in enumerate(city_vals):
        if k < len(city_vals) - 1:
            city_change += abs(city_vals[k] - city_vals[k+1])
            if maxv < city_vals[k+1]:
                maxv = city_vals[k+1]
            if minv > city_vals[k+1]:
                minv = city_vals[k+1]
        day_change[col] = city_change
        max_temp[col] = maxv
        min_temp[col] = minv
        var_temp[col] = max_temp[col] - min_temp[col]
        
def keywithmaxval(d):
    return max(d, key=lambda k: d[k])

max_city = keywithmaxval(day_change)
print ('The city that has the highest change is ' + str(max_city) + ' with ' + \
       str(day_change[max_city]) + ' of chnages within the last month')

max_change_city = keywithmaxval(var_temp)
print ('The city that has the highest variance is ' + str(max_change_city) + ' with ' + \
       str(var_temp[max_change_city]) + ' of range within the last month')

## Visually Inspect the Data
import matplotlib.pyplot as plt
plt.bar(range(len(day_change)), day_change.values(), align='center')
plt.xticks(range(len(day_change)), list(day_change.keys()))
plt.show()
plt.savefig('bar_temp_change.png')

plt.bar(range(len(var_temp)), var_temp.values(), align='center')
plt.xticks(range(len(var_temp)), list(var_temp.keys()))
plt.show()
plt.savefig('bar_temp_range.png')