import pandas as pd
from pandas.tseries.offsets import MonthBegin

from acaParser import process_images
from neoParser import process_neo
from calipsoParser import process_sat
import numpy as np

import matplotlib.pyplot as plt


dir_anom = '/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/temp_anom'
dir_coral = '/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/coral_snapshot'
dir_neo = '/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/neo_data'
dir_calipso = '/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/2021272175244_62409/'

bound_top = 26.405982971191
bound_left = -82.882919311523
bound_right = -79.850692749023
bound_bottom = 24.208717346191

df_anom = process_images(dir_anom, 'temp_anom', bound_top, bound_left, bound_bottom, bound_right, False)
df_coral = process_images(dir_coral, 'coral_algae', bound_top, bound_left, bound_bottom, bound_right, True)
df_neo = process_neo(dir_neo, bound_top, bound_left, bound_bottom, bound_right)
df_sat = process_sat(dir_calipso)


def aggregate_data(df_main, metric):
    df = df_main
    # df = df[df.temp_anom > 0]
    df['Long'] = np.round(df['Long'], 2)
    df['Lat'] = pd.to_numeric(df.Lat)
    df['Lat'] = np.round(df.Lat, 2)
    return df.groupby(['Lat', 'Long', 'Date']).agg(X=(metric, 'median')).reset_index()


df_anom_copy = aggregate_data(df_anom, 'temp_anom')
df_coral_copy = aggregate_data(df_coral, 'coral_algae')
df_neo_copy = aggregate_data(df_neo, 'chlorophyll')

# aggregate satellite data
df_sat['Long'] = df_sat['Long'].astype(float).round(2)
df_sat['Lat'] = df_sat.Lat.astype(float).round(2)


test_set_full = df_anom[df_anom.Date == '2020-02-21']
plt.scatter(x=test_set_full.Lat, y=test_set_full.Long, c=test_set_full.temp_anom)
test_set_abbr = aggregate_data(test_set_full, 'temp_anom')
plt.scatter(x=test_set_abbr.Lat, y=test_set_abbr.Long, c=test_set_abbr.X)

df_anom_copy['Date'] = pd.to_datetime(df_anom_copy['Date']) - MonthBegin(1)
df_merged = df_anom_copy.merge(df_neo_copy, on=['Lat', 'Long', 'Date'], how='outer')
df_merged = df_merged.merge(df_sat, on=['Lat', 'Long', 'Date'], how='outer')

df_sample = df_merged[df_merged.Date == '2020-06-01']
df_sample.drop(['index'], inplace=True)