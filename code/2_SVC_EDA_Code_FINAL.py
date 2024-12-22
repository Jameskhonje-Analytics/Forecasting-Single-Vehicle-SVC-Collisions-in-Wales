# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 21:12:40 2024

@author: khonj
"""
#%%
# ==========================================================================================
# Loading packages
# ==========================================================================================
# Importing necessary packages

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
#import glob    # For reading multiple files

#! pip install sorted-months-weekdays
#! pip install sort-dataframeby-monthorweek

import datetime
import calendar

from sorted_months_weekdays import *
from sort_dataframeby_monthorweek import *

# For geoprocessing

from datetime import datetime 
#import geopandas as gpd

# Set background of plots to white

sns.set_style(style='white')
#%%
# ==========================================================================================
# <h2>Loading data for Temporal and Spatial Analysis</h2>
# ==========================================================================================

# df = df_col_rearranged.copy()

# Reading data saved to disk with labels for Analysis
#df = pd.read_csv("..\data\Cleaned_Project_Data.csv")

# Create dataset for Spatial analysis only
#df_spatial_Analysis = pd.read_csv("..\data\Cleaned_Project_Data.csv")

# path = "D:\\OneDrive\\_MSc_DScience_Term_1\\0_Dissertation\\data\\Cleaned_Project_Data.csv"

# df_spatial_Analysis= pd.read_csv(path)



path1 = "..\\data\\SVC_cleaned_Wales_Data_2010_2022.csv"

df= pd.read_csv(path1)
df


#%%

# ==========================================================================================
# 
# ==========================================================================================
#%%
# Convert date to date format

df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y")


# Create time variables Month, Weekday

df['date'] = pd.to_datetime(df['date'])
df['month'] = pd.to_datetime(df['date']).dt.strftime('%b')
df['Weekday'] = pd.to_datetime(df['date']).dt.strftime('%a')
df['Mont_yr'] = pd.to_datetime(df['date']).dt.strftime('%b-%y')

# Convert multiple columns to strings
df[['PoliceForce', 'speed_limit','age_of_driver']] = df[['PoliceForce', 'speed_limit','age_of_driver']]. astype(str)

# ==========================================================================================
# 
# ==========================================================================================

df.describe()
df.info()
#%%
# ==========================================================================================
# 
# ==========================================================================================

# Rearrange features in our dataframe

df = df.iloc[:,[0,1,27,28,29,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]]

#%%
# ==========================================================================================
# Create Categorical variables
# ==========================================================================================

# Reordering time variables in right order
df_eda = df.copy()

# Order months in the right order

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df_eda['month'] = pd.Categorical(df_eda['month'], categories=months, ordered=True)
df_eda.sort_values(by='date',inplace=True)


# Order week days in the right order
cats = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
df_eda['Weekday'] = pd.Categorical(df_eda['Weekday'], categories=cats, ordered=True)
df_eda.sort_values(by='date',inplace=True)

df_eda.PoliceForce = df_eda.PoliceForce.astype('category')

# Convert multiple columns to category type
cols = ['PoliceForce', 'severity', 'Road_class', 'road_type','speed_limit','junction_detail','season','light_conditions','weather','road surface conditions','urban or rural','vehicle_type']
df_eda[cols] = df_eda[cols].astype('category')



df_eda

#%%
# ==========================================================================================
# Line chart of single vehicle collisions
# ==========================================================================================

# Group the data by year and sum the counts
svc_dat = df_eda.groupby('year')['count'].sum().reset_index()

sns.set_style('white')
plt.figure(figsize=(12, 6))
plt.plot(svc_dat['year'], svc_dat['count'], marker='o', linestyle='-', color='b')

# Add title and labels
# plt.title('Number of Accidents by Year')
plt.xlabel('Year', fontweight= 'bold', fontsize = 12)
plt.ylabel('Collisions', fontweight= 'bold', fontsize = 12)

# Set y-axis to begin from zero 
plt.ylim(bottom = 0, top = 600)

# Remove top and right borders
sns.despine(left=False, bottom=False)

# Show the plot
plt.grid(True)
plt.show()



#%%

# ==========================================================================================
# Heatmap of collisions by month by year
# ==========================================================================================
# First pivot data by Month

df_pivot = df_eda.groupby(['year','month'])['count'].sum().reset_index()
df_pivot

month_pivot=df_pivot.pivot_table(values='count',index='month',columns='year')
# month_pivot

# Heatmap showing collisions distribution by month and year

sns.set_theme(rc={'figure.figsize':(12,6)})

sns.heatmap(month_pivot, annot=True, fmt=".0f", 
            cmap='Blues',
            linecolor='white',
            #annot=True,
            annot_kws={'size': 9},
            linewidth=0.5)

# plt.title("Heatmap of collisions by Month and Year")
plt.ylabel("Month", fontweight= 'bold', fontsize = 11)
plt.xlabel('Year', fontweight= 'bold', fontsize = 11)

plt.show()

#%%
#plt.savefig("HeatmapA.png") 

# Feature selection  

df_eda = df_eda.drop(columns=['eastings','northings','lon','lat','AccRef','LA_number','time','date','PoliceForce','sex_of_driver'])


# ==========================================================================================
# Spatial Analysis of collisions involving a single vehicle
# ==========================================================================================

#df.groupby('month')['count'].sum().reset_index().sort_values(by='count')
#%%

# Create a variable for month count
month_distribution = df_eda.groupby('month')['count'].count().reset_index().sort_values(by='count')

sns.set_theme(rc={'figure.figsize':(8,4)},palette='Blues_d')
sns.set_style(style='white')
sns.barplot(y=month_distribution['count']/12, x='month', data= month_distribution)

plt.title("Number of collisions per month")
plt.ylabel("Average monthly collisions" , fontweight= 'bold', fontsize = 12)
plt.xlabel("Number of collisions", fontweight= 'bold', fontsize = 12)

sns.despine(left=False, bottom=False)
plt.show()

#plt.savefig("Monthly collisions.png")

#%%


# Another chart for month. Create monthly data 

months_Data = df_eda.groupby('year')['month'].value_counts().reset_index()

sns.set_style(style='white')
sns.boxplot(data = months_Data, x = 'month', y = 'count', palette= 'rainbow',
            showmeans=True,
            meanprops={"marker": ".",          # To show the mean value
                       "markersize": "7",
                       "markerfacecolor": "red", # Change marker face color 
                       "markeredgecolor": "red"
                       })
sns.set_theme(rc={'figure.figsize':(12,6)})
plt.title("Number of collisions per month")
plt.xlabel("month")
plt.ylabel("Number of collisions", fontweight= 'bold', fontsize = 12)


plt.grid()

sns.despine(left=False, bottom=False)
plt.show()
#%%
# ==========================================================================================
# Urban or Rural area
# ==========================================================================================

sns.countplot(data=df_eda,x='urban or rural',palette='Blues_d')

sns.set_theme(rc={'figure.figsize':(12,6)})
sns.set_style(style='white')
plt.title("Single vehicle collision by area")
plt.ylabel("Number of collisions", fontweight= 'bold', fontsize = 12)
plt.xlabel(" ")

sns.despine(left=False, bottom=False)
plt.grid()
plt.show()

#%%
# Creating new dataframe to plot box plot for months and total collisions per month
# =====================================================================================

# Set the theme to white
# sns.set_theme(style="white")
sns.set_theme(rc={'figure.figsize':(12,6)})
sns.set_style(style='white')
df_box = df_eda.groupby(['year','urban or rural'])['month'].value_counts().reset_index()

sns.boxplot(x='month', y='count', data=df_box, palette='rainbow',
            hue ="urban or rural",
            showmeans=True,
            meanprops={"marker": ".",          # To show the mean value
                       "markersize": "7",
                       "markerfacecolor": "red", # Change marker face color 
                       "markeredgecolor": "red"
                       })

plt.title("Collisions by Month")
plt.ylabel("Number of collisions", fontweight= 'bold', fontsize = 12)
plt.xlabel("Months", fontweight= 'bold', fontsize = 12)
# turn on the grid
plt.grid()
# Remove Top and Right borders
sns.despine(left=False, bottom=False)
plt.legend(facecolor='gray', framealpha=1, loc='upper left')
plt.show()

#plt.savefig("Month by Rural_Urban.png")
#%%
# ==========================================================================================
# Collision analysis by Week day Average
# ==========================================================================================

Week_distibution = df_eda.groupby('Weekday')['count'].sum().reset_index().sort_values(by='count')


sns.barplot(y=round(Week_distibution['count']/48,3), x='Weekday', data=Week_distibution,palette='Blues_d')

sns.set_theme(rc={'figure.figsize':(12,6)})

plt.title("Collisions by Weekday")
plt.ylabel("Average number of collisions", fontweight= 'bold', fontsize = 12)
plt.xlabel("Months", fontweight= 'bold', fontsize = 12)

sns.set_style(style='white')

# Remove Top and Right borders
sns.despine(left=False, bottom=False)

plt.show()

#plt.savefig("Weekly collisions.png")

#%%

df_box_weekday = df_eda.groupby(['year','urban or rural'])['Weekday'].value_counts().reset_index()

sns.set_theme(rc={'figure.figsize':(12,6)})
sns.set_style(style='white')
sns.boxplot(y='Weekday', x='count', 
            data=df_box_weekday, 
            palette='rainbow',
            hue ="urban or rural",
            showmeans=True,
            meanprops={"marker": ".",          # To show the mean value
                       "markersize": "7",
                       "markerfacecolor": "red", # Change marker face color 
                       "markeredgecolor": "red"
                       })

plt.title("Collisions by Weekday and Area")
plt.ylabel("Week day", fontweight= 'bold', fontsize = 12)
plt.xlabel("Number of collisions", fontweight= 'bold', fontsize = 12)

# turn on the grid
plt.grid()

plt.legend(facecolor='gray', framealpha=1, loc='upper right')
# Remove Top and Right borders
sns.despine(left=False, bottom=False)

plt.show()

#plt.savefig("Weekly by Rural_Urban.png")
# ==========================================================================================
#%%

import seaborn as sns
import pandas as pd

# order data
ordered_categories = df_eda['Road_class'].value_counts(ascending=False).index
sns.set_style(style='white')
sns.countplot(data=df_eda, 
              x='Road_class', 
              order=ordered_categories, 
              palette='Blues_d')

#sns.countplot(data=df_eda,x='Road_class',palette='Blues_d')

sns.set_theme(rc={'figure.figsize':(12,6)})

plt.title("Single vehicle collision by Road Class")
plt.ylabel("Number of collisions", fontweight= 'bold', fontsize = 12)
plt.xlabel(" ")

sns.despine(left=False, bottom=False)

plt.show()




#%%
# ==========================================================================================

# ==========================================================================================
#  Analysis of Collisions by Season
# ==========================================================================================

# Its important to account for seasonality impact on collisions to interpret data correctly. 
# This section analyses SVC seasonal pattern for the whole period and captures the trends overtime. 
# Based on the outputs most SVC are reported during summer followed by AUtumn.

df_eda.groupby('season')['count'].sum().plot(kind="bar", figsize = (10, 6))


sns.set_theme(rc={'figure.figsize':(12,6)})

plt.title("Number of collisions by Season")
plt.xlabel("Season", fontweight= 'bold', fontsize = 12)
plt.ylabel("Number of collisions", fontweight= 'bold', fontsize = 12)

# Remove Top and Right borders
sns.despine(left=False, bottom=False)

plt.show()


#%%
df_box_season = df_eda.groupby(['year','urban or rural'])['season'].value_counts().reset_index()

sns.set_theme(rc={'figure.figsize':(12,6)})

sns.set_style(style='white')

sns.boxplot(y='season', x='count', 
            data=df_box_season, 
            palette='rainbow',
            hue ="season",
            showmeans=True,
            meanprops={"marker": ".",          # To show the mean value
                       "markersize": "7",
                       "markerfacecolor": "red", # Change marker face color 
                       "markeredgecolor": "red"
                       })

# Remove Top and Right borders
sns.despine(left=False, bottom=False)

plt.xlabel("Number of collisions", fontweight= 'bold', fontsize = 12)
plt.ylabel("Season", fontweight= 'bold', fontsize = 12)
plt.show()
#plt.savefig("SeasonA by Rural_Urban.png")

#%%
df_box_season = df_eda.groupby(['year','urban or rural'])['season'].value_counts().reset_index()

sns.set_theme(rc={'figure.figsize':(12,6)})
sns.set_style(style='white')
sns.boxplot(y='season', x='count', 
            data=df_box_season, 
            palette='rainbow',
            hue ="urban or rural",
            showmeans=True,
           meanprops={"marker": ".",          # To show the mean value
                      "markersize": "7",
                      "markerfacecolor": "red", # Change marker face color 
                      "markeredgecolor": "red"
                      })

# Remove Top and Right borders
sns.despine(left=False, bottom=False)

plt.xlabel("Number of collisions", fontweight= 'bold', fontsize = 12)
plt.ylabel("Season", fontweight= 'bold', fontsize = 12)
plt.legend(facecolor='gray', framealpha=1, loc='upper right')
plt.grid()
plt.show()
#plt.savefig("Season by Rural_Urban.png")

#%%
df_box_season = df_eda.groupby(['year'])['urban or rural'].value_counts().reset_index()

sns.set_theme(rc={'figure.figsize':(12,6)})
sns.set_style(style='white')
sns.boxplot(y='urban or rural', x='count', 
            data=df_box_season, 
            palette='rainbow',
            hue ="urban or rural",
            showmeans=True,
            meanprops={"marker": ".",          # To show the mean value
                       "markersize": "7",
                       "markerfacecolor": "red", # Change marker face color 
                       "markeredgecolor": "red"
                       })

sns.despine(left=False, bottom=False)
plt.xlabel("Number of collisions", fontweight= 'bold', fontsize = 12)
plt.ylabel("Area", fontweight= 'bold', fontsize = 12)
plt.grid()
plt.show()
#plt.savefig("Month by Rural_Urban.png")

#%%
# ==========================================================================================
# Collisions by Time of the Day
# ==========================================================================================

df_eda.groupby('hour')['count'].sum().plot(kind="bar", figsize = (10, 6))

sns.set_theme(rc={'figure.figsize':(12,6)})
sns.set_style(style='white')
plt.title("Number of collisions per hour")
plt.xlabel("Hour", fontweight= 'bold', fontsize = 12)
plt.ylabel("Number of collisions", fontweight= 'bold', fontsize = 12)

# Remove Top and Right borders
sns.despine(left=False, bottom=False)

plt.show()
#%%
# ==========================================================================================
# Boxplot of Collisions by Time of the Day
# ==========================================================================================

time_Data = df_eda.groupby(['year','hour'])['count'].sum().reset_index()

sns.set_theme(rc={'figure.figsize':(12,6)})
sns.set_style(style='white')
sns.boxplot(data = time_Data, x = 'hour', y='count', palette='rainbow',
            showmeans=True,
            meanprops={"marker": ".",          # To show the mean value
           "markersize": "7",
           "markerfacecolor": "red", # Change marker face color 
           "markeredgecolor": "red"
           })
plt.title("Number of collisions per hour")
plt.xlabel("Hour", fontweight= 'bold', fontsize = 12)
plt.ylabel("Number of collisions", fontweight= 'bold', fontsize = 12)

plt.grid()
# Remove Top and Right borders
sns.despine(left=False, bottom=False)

plt.show()




# ==========================================================================================
# 
# ==========================================================================================













