import pandas as pd
from pandas.tseries.offsets import MonthBegin

# from acaParser_v2 import process_images
from neoParser import process_neo
from calipsoParser import process_sat
import geopandas
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
import shapely

dir_anom = '/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/temp_anom'
dir_coral = '/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/coral_snapshot'
dir_neo = '/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/neo_data'
dir_calipso = '/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/2021272175244_62409/'
dir_geo = '/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/'

bound_top = 26.405982971191
bound_left = -82.882919311523
bound_right = -79.850692749023
bound_bottom = 24.208717346191


df_neo = process_neo(dir_neo, bound_top, bound_left, bound_bottom, bound_right)
df_sat = process_sat(dir_calipso)

'''def aggregate_data(df_main, metric):
    df = df_main
    # df = df[df.temp_anom > 0]
    df['Long'] = np.round(df['Long'], 4)
    df['Lat'] = pd.to_numeric(df.Lat)
    df['Lat'] = np.round(df.Lat, 4)
    return df.groupby(['Lat', 'Long', 'Date']).agg(X=(metric, 'median')).reset_index()'''


def create_gdf(df):
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Lat, df.Long),
                                 crs="+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs")
    gdf.drop(['Lat', 'Long'], axis=1, inplace=True)
    return gdf


gdf_neo = create_gdf(df_neo)
gdf_sat = create_gdf(df_sat)



# gdf_anom.to_csv('temp_anom.csv')
# gdf_coral.to_csv('coral.csv')
# gdf_neo.to_csv('neo_data.csv')
# gdf_sat.to_csv('calipso.csv')


# https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point-in-other-dataframe
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


####### Spatial Grid Test
# https://james-brennan.github.io/posts/fast_gridding_geopandas/
