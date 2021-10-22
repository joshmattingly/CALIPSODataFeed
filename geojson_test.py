import urllib
import os
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, shape
import shapely.speedups

import calipsoParser
from neoParser import process_neo
from scipy.spatial import cKDTree
from pyspark.sql.functions import *
from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType,DecimalType
from pyspark.sql.functions import lit, pandas_udf, PandasUDFType


gdf_benth = gpd.read_file('/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/'
                                'Benthic-Map/benthic.shp')

gdf_geo = gpd.read_file('/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/'
                              'Geomorphic-Map/geomorphic.shp')

shapely.speedups.enable()

# dir_neo = '/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/neo_data'
bound_top = 26.405982971191
bound_left = -82.882919311523
bound_right = -79.850692749023
bound_bottom = 24.208717346191


# df_neo = process_neo(dir_neo, bound_top, bound_left, bound_bottom, bound_right)


def create_gdf(df):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Lat, df.Long),
                                 crs="+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs")
    gdf.drop(['Lat', 'Long'], axis=1, inplace=True)
    return gdf


def ckdnearest(gdA, gdB):

    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ],
        axis=1)

    return gdf


# gdf_neo = create_gdf(df_neo)


