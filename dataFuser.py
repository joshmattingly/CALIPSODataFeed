import pandas as pd
from pandas.tseries.offsets import MonthBegin

from acaParser import process_images
from neoParser import process_neo
from calipsoParser import process_sat
from translateCoords import getlonglat, getxy
import geopandas
import numpy as np
from sqlalchemy import create_engine

import matplotlib.pyplot as plt
import utm

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
    df['Long'] = np.round(df['Long'], 4)
    df['Lat'] = pd.to_numeric(df.Lat)
    df['Lat'] = np.round(df.Lat, 4)
    return df.groupby(['Lat', 'Long', 'Date']).agg(X=(metric, 'median')).reset_index()


df_anom['Date'] = pd.to_datetime(df_anom['Date']) - MonthBegin(1)


def create_gdf(df):
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Lat, df.Long))
    gdf.drop(['Lat', 'Long'], axis=1, inplace=True)
    return gdf


gdf_anom = create_gdf(df_anom)
gdf_coral = create_gdf(df_coral)
gdf_neo = create_gdf(df_neo)
gdf_sat = create_gdf(df_sat)

engine = create_engine("postgresql://PostGIS:5432/josh")
gdf_coral.to_postgis("coral", engine)
gdf_anom.to_postgis("temp_anom", engine)
gdf_anom.to_postgis("temp_anom", engine)
