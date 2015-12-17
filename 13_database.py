# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:42:55 2015

@author: Michael Lin_2
"""

import sqlite3 as lite
import pandas as pd

con = lite.connect('getting_started.db')

cities = (('New York City', 'NV'),
          ('Boston', 'MA'),
          ('Chicago', 'IL'),
          ('Miami', 'FL'),
          ('Seattle', 'WA'),
          ('Portland', 'OR'),
          ('San Francisco', 'CA'),
          ('Los Angeles', 'CA'),
          ('Washington', 'DC'),
          ('Houston', 'TX'),
          ('Las Vegas', 'NV'),
          ('Atlanta', 'GA'))

weather = (('New York City', 2013, 'July', 'January', 62),
           ('Boston', 2013, 'July', 'January', 59),
           ('Chicago', 2013, 'July', 'January', 84),
           ('Miami', 2013, 'August', 'January', 84),
           ('Seattle', 2013, 'July', 'January', 61),
           ('Portland', 2013, 'July', 'December', 63),
           ('San Francisco', 2013, 'September', 'December', 64),
           ('Los Angeles', 2013, 'September', 'December', 75),
           ('Washington', 2013, 'July', 'January', 60),
           ('Houston', 2013, 'July', 'January', 50),
           ('Las Vegas', 2013, 'July', 'December', 49),
           ('Atlanta', 2013, 'July', 'January', 43),
           ('Dallas', 2013, 'July', 'January', 77))

con = lite.connect('getting_started.db')

# Inserting rows by passing tuples to `execute()`
with con:
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS cities")
    cur.execute("DROP TABLE IF EXISTS weather")
    cur.execute("CREATE TABLE cities (name text, state text)")
    cur.execute("CREATE TABLE weather (city text, year integer, warm_month text, cold_month text, average_high integer)")
    cur.executemany("INSERT INTO cities VALUES(?,?)", cities)
    cur.executemany("INSERT INTO weather VALUES(?,?,?,?,?)", weather)

# Select only July as warm_month
with con:
  cur = con.cursor()
  cur.execute("SELECT city, state, warm_month FROM weather INNER JOIN cities ON city = name WHERE warm_month = 'July'")

  rows = cur.fetchall()
  cols = [desc[0] for desc in cur.description]
  df = pd.DataFrame(rows, columns = cols)