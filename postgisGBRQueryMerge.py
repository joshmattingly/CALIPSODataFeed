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


'''server = SSHTunnelForwarder(
    ('pace-ice.pace.gatech.edu', 22),
    ssh_username=input("Username: "),
    ssh_password=getpass.getpass(prompt='Password: ', stream=None),
    remote_bind_address=('127.0.0.1', 5432)
)
server.start()

local_port = str(server.local_bind_port)'''
engine = create_engine('postgresql://{}@{}:{}/{}'.format("jmattingly31", "127.0.0.1",
                                                         5432, "coral_data"))
# gdf_benth = geopandas.read_file('GBR/Benthic-Map/benthic.shp')
# gdf_geo = geopandas.read_file('GBR/Geomorphic-Map/geomorphic.shp')

print("Importing CALIPSO data")
df_calipso = geopandas.read_postgis("""
SELECT * FROM manta_200
""", con=engine, geom_col='geom')

print("Importing AIM data")
df_manta = geopandas.read_postgis("""
SELECT * FROM manta_tow_data
""", con=engine, geom_col='geom')

df_calipso['calipso_date'] = pd.to_datetime(df_calipso['calipso_date'])
df_calipso['manta_date'] = pd.to_datetime(df_calipso['manta_date'])

print("Importing NEO Data")
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
df_manta_dates = df_manta['sample_date'].unique()

gdf_main = None
print('Processing NEO Dates')
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

print('Processing Manta Tow Data')
for date in df_manta_dates:
    print(date)
    try:
        temp_calipso = df_calipso[df_calipso.manta_date == date]
        temp_manta = df_manta[df_manta.sample_date == date]
        temp_gdf = ckdnearest(temp_calipso, temp_manta)
        if gdf_main is None:
            gdf_main = temp_gdf
        else:
            gdf_main = gdf_main.append(temp_gdf)
    except:
        print("Error processing {}".format(date))
        continue

# gdf_benth['geom'] = gdf_benth.geometry.centroid
# gdf_benth['geom'] = gdf_benth.apply(lambda row: Point(row.geom.x, row.geom.y), axis=1)

# gdf_geo['geom'] = gdf_geo.geometry.centroid
# gdf_geo['geom'] = gdf_geo.apply(lambda row: Point(row.geom.y, row.geom.x), axis=1)


# gdf_main.drop(['dist'], axis=1, inplace=True)
# gdf_main_benth = ckdnearest(gdf_main, gdf_benth)
# gdf_main_benth.reset_index(inplace=True)
# gdf_total = ckdnearest(gdf_main_benth, gdf_geo)

# gdf_main_benth.to_csv('florida_200_full.csv')

# gdf_main_benth.to_sql('florida_100_full', engine, index=False, if_exists='replace')
gdf_classified = gdf_main[gdf_main.dist < 0.5]
gdf_classified.to_csv('manta_200_classified.csv')
