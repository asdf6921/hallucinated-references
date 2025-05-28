import pandas as pd
import numpy as np

"""
takes airports.csv and gets 1000 rows, and the columns are name, elevation_ft, iso_country, iso_region, continent
and map the last 3 to actual names using countries.csv, regions.csv
"""
# Load airports.csv
airports = pd.read_csv('airports.csv')

# Select required columns and get 1000 random rows
selected_columns = ['name', 'elevation_ft', 'iso_country', 'iso_region', 'continent']
airports_subset = airports[selected_columns]

all_continents = airports['continent'].unique()
mappings = {'NA': 'North America',
            'EU': 'Europe',
            'AS': 'Asia',
            'AF': 'Africa',
            'SA': 'South America'}
airports_subset.to_csv('airports_subset.csv', index=False)

"""
Queries: 
Call below for each of (5) ['North America', 'Europe', 'Asia', 'Africa', 'South America']: you would list 200
List {} existing world airports.  Just output the exact official airport names and nothing else. Output format should be <num.> <name>

For this world airport: {}, what is the elevation in feet? Output format should be <elevation_ft>

For this world airport: {}, what is the country it is in? Output format should be <country_name> where <country_name> is the exact full name of the country

For this world airport: {}, what is the continent it is in? Output format should be <continent_name> where <continent_name> is one of these 5 options: North America, Europe, Asia, Africa, South America
"""