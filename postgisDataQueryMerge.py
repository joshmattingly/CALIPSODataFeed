import pandas as pd
import numpy as np
import getpass
import geopandas
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder
from scipy.spatial import cKDTree
from shapely.geometry import Point
from geopandas import gpd
from geoalchemy2.types import Geometry


gdf_benth = geopandas.read_file('/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/Benthic-Map/benthic.shp')
gdf_geo = geopandas.read_file('/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/Geomorphic-Map/geomorphic.shp')


def create_gdf(df):
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Lat, df.Long),
                                 crs="+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs")
    gdf.drop(['Lat', 'Long'], axis=1, inplace=True)
    return gdf


server = SSHTunnelForwarder(
    ('pace-ice.pace.gatech.edu', 22),
    ssh_username=input("Username: "),
    ssh_password=getpass.getpass(prompt='Password: ', stream=None),
    remote_bind_address=('127.0.0.1', 5432)
)
server.start()

local_port = str(server.local_bind_port)
engine = create_engine('postgresql://{}@{}:{}/{}'.format("jmattingly31", "127.0.0.1",
                                                         local_port, "coral_data"))

df_calipso = geopandas.read_postgis("""
SELECT
       c."Date" as calipso_date,
       c."Lat",
       c."Long",
       c."578",
       c."579",
       c."580",
       c."581",
       c."582",
       c."geom",
       n."Date" as neo_file_date
FROM (SELECT * FROM calipso_data as c WHERE c."Date" >= '2018-01-01') c
CROSS JOIN LATERAL (
  SELECT n."Date"
  FROM neo_dates as n
  WHERE c."Date" <= n."Date"
  ORDER BY n."Date"
  LIMIT 1
) AS n
""", con=engine, geom_col='geom')

df_calipso['calipso_date'] = pd.to_datetime(df_calipso['calipso_date'])
df_calipso['neo_file_date'] = pd.to_datetime(df_calipso['neo_file_date'])

df_neo = geopandas.read_postgis("SELECT * FROM neo_data", con=engine, geom_col='geom')
df_neo['Date'] = pd.to_datetime(df_neo['Date'])


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


df_neo_dates = df_neo['Date'].unique()

gdf_main = None

for date in df_neo_dates:
    print(date)
    try:
        temp_calipso = df_calipso[df_calipso.neo_file_date == date]
        temp_neo = df_neo[df_neo.Date == date]
        temp_gdf = ckdnearest(temp_calipso, temp_neo)
        if gdf_main is None:
            gdf_main = temp_gdf
        else:
            gdf_main = gdf_main.append(temp_gdf)
    except:
        print("error processing {}".format(date))
        continue

gdf_benth['geometry'] = gdf_benth.geometry.centroid
gdf_benth['geometry'] = gdf_benth.apply(lambda row: Point(row.geometry.y, row.geometry.x), axis=1)

gdf_geo['geometry'] = gdf_geo.geometry.centroid
gdf_geo['geometry'] = gdf_geo.apply(lambda row: Point(row.geometry.y, row.geometry.x), axis=1)

gdf_main_benth = ckdnearest(gdf_main, gdf_benth)
gdf_main_benth.columns = ['calipso_date', '578', '579', '580', '581', '582', 'neo_file_date',
                          'geometry', 'chlorophyll', 'Date', 'dist', 'benth_class', 'benth_dist']

gdf_total = ckdnearest(gdf_main_benth, gdf_geo)
gdf_total.columns = ['calipso_date', '578', '579', '580', '581', '582', 'neo_file_date',
'geometry', 'chlorophyll', 'neo_date', 'neo_dist', 'benth_class', 'benth_dist',
'geo_class', 'geo_dist']

gdf_total.to_csv('florida_full.csv')

gdf_total.to_sql('florida_full', engine, index=False, if_exists='replace')
gdf_classified = gdf_total[(gdf_total.benth_dist < 0.1) | (gdf_total.geo_dist < 0.1)]
