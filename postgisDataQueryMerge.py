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
gdf_benth = geopandas.read_file('/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/Benthic-Map/benthic.shp')
gdf_geo = geopandas.read_file('/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/Geomorphic-Map/geomorphic.shp')

df_calipso = geopandas.read_postgis("""
SELECT * FROM florida_100
""", con=engine, geom_col='geom')

df_calipso['calipso_date'] = pd.to_datetime(df_calipso['calipso_date'])
df_calipso['neo_file_date'] = pd.to_datetime(df_calipso['neo_file_date'])

df_neo = geopandas.read_postgis("SELECT * FROM neo_data", con=engine, geom_col='geom')
df_neo['Date'] = pd.to_datetime(df_neo['Date'])


def ckdnearest(gdA, gdB):
    nA = np.array(list(gdA.geom.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geom.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geom").reset_index(drop=True)
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
        print("Error processing {}".format(date))
        continue

gdf_benth['geom'] = gdf_benth.geometry.centroid
gdf_benth['geom'] = gdf_benth.apply(lambda row: Point(row.geometry.y, row.geometry.x), axis=1)

gdf_geo['geom'] = gdf_geo.geometry.centroid
gdf_geo['geom'] = gdf_geo.apply(lambda row: Point(row.geometry.y, row.geometry.x), axis=1)


gdf_main.drop(['dist'], axis=1, inplace=True)
gdf_main_benth = ckdnearest(gdf_main, gdf_benth)
gdf_main_benth.reset_index(inplace=True)
# gdf_total = ckdnearest(gdf_main_benth, gdf_geo)

gdf_main_benth.to_csv('florida_100_full.csv')

# gdf_main_benth.to_sql('florida_100_full', engine, index=False, if_exists='replace')
gdf_classified = gdf_main_benth[gdf_main_benth.dist < 0.5]
gdf_classified.to_csv('florida_100_classified.csv')


gdf_class_test = gdf_classified.copy()
gdf_class_test = gdf_class_test[gdf_class_test['class'] != 'Seagrass']

dummies = gdf_class_test['class'].str.get_dummies()
dummies.columns = ['is_' + col for col in dummies.columns]
gdf_class_test = pd.concat([gdf_class_test, dummies], axis=1)

gdf_class_test.drop(['index', 'calipso_date', 'Date', 'Long', 'Lat', 'geom', 'geometry', 'dist',
                     'neo_file_date', 'Latitude', 'Longitude', 'is_Rock', 'is_Rubble', 'is_Sand', 'class'],
                    axis=1, inplace=True)

