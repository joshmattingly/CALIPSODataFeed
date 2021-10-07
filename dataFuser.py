import pandas as pd

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
    df['Long'] = np.round(df['Long'], 3)
    df['Lat'] = pd.to_numeric(df.Lat)
    df['Lat'] = np.round(df.Lat, 3)
    return df.groupby(['Long', 'Lat', 'Date']).agg(X=(metric, 'median')).reset_index()


df_anom_copy = aggregate_data(df_anom, 'temp_anom')
df_coral_copy = aggregate_data(df_coral, 'coral_algae')
df_neo_copy = aggregate_data(df_neo, 'chlorophyll')


test_set_full = df_anom[df_anom.Date == '2020-02-21']
plt.scatter(x=test_set_full.Lat, y=test_set_full.Long, c=test_set_full.temp_anom)
test_set_abbr = aggregate_data(test_set_full, 'temp_anom')
plt.scatter(x=test_set_abbr.Lat, y=test_set_abbr.Long, c=test_set_abbr.X)
