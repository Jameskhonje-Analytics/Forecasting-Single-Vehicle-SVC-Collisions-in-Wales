# Importing necessary packages


#%%
import pandas as pd
import numpy as np

#%%
# ===============================================================================
# Data Creation for Group Project - Individual
# ===============================================================================

df_col = pd.read_csv("..\\data\\Road_safety_data\\dft-road-casualty-statistics-collision-1979-latest-published-year.csv")
df_veh = pd.read_csv("..\\data\Road_safety_data\\dft-road-casualty-statistics-vehicle-1979-latest-published-year.csv")


#%%

# Select data from 2010 onwards instead of 1970 from Collisions/casualties and vehicles tables

# ==============================
# Collisions data Cleaning
# ==============================

df_collision = df_col[df_col["accident_year"] >= 2010]

# Filter data for Welsh Police Forces only
df_collision =df_collision[df_collision["police_force"].isin([60, 61, 62, 63])]

# Convert speed limit variable from float to integer and AgeGroup to dummy and int

df_collision['speed_limit'] = np.int64(df_collision['speed_limit'])

df_collision['speed_limit'].replace([10,20,30,40,50,60,70], [1,2,3,4,5,6,7], inplace=True)

# Rename column names for collision data

df_collision.rename(columns={'accident_index': 'Acc_ind', 'accident_year': 'Year', 'accident_reference': 'AccRef', 'day_of_week': 'Week day',
                             'location_easting_osgr': 'eastings', 'location_northing_osgr': 'northings', 'police_force': 'Police Force', 
                             'number_of_casualties': 'Casualties','accident_severity': 'Severity', 'first_road_class': 'Road class', 
                             'weather_conditions': 'Weather', 'local_authority_highway': 'LA_number', 'local_authority_district': 'LA', 
                             'local_authority_ons_district': 'LA ONS', 'urban_or_rural_area': 'Urban or rural'}, inplace=True)   # Rename the column

df_collision.head().T

#%%
# Create a copy of our datasets

col_dataA = df_collision.copy()

# ===============================================================================
# Select data where only a single vehicle was involved in the collision and only KSI cases were reported
# ===============================================================================

col_dataA.dtypes

col_dataB = col_dataA[col_dataA["number_of_vehicles"] ==1 & (col_dataA["Severity"].isin([1,2]))]


# Drop Columns from our dataframe from 36

col_data = col_dataB.drop(columns=['Acc_ind','pedestrian_crossing_physical_facilities', 'special_conditions_at_site','LA ONS', 'LA','number_of_vehicles',
                           'did_police_officer_attend_scene_of_accident','trunk_road_flag','first_road_number','lsoa_of_accident_location','road_surface_conditions',
                          'pedestrian_crossing_human_control','carriageway_hazards','second_road_class','second_road_number','junction_control'], axis = 1)


df_collisionsdata = col_data.copy()

# Create Month, count & hour variables for each collision in the table 

df_collisionsdata['date'] = pd.to_datetime(df_collisionsdata['date'], format="%d/%m/%Y")
df_collisionsdata['month'] = pd.to_datetime(df_collisionsdata['date']).dt.month
df_collisionsdata['count'] = 1
df_collisionsdata['hour'] = df_collisionsdata['time'].str[:2]

df_collisionsdata.dtypes


df = df_collisionsdata[['date','count']]

df.to_csv("..\data\Wales_project_data.csv", index=False)



df.head()

#   =================== End of data generation =========================================================

#%%

