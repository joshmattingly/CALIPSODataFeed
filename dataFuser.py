import pandas as pd
from pandas.tseries.offsets import MonthBegin

from acaParser_v2 import process_images
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

df_anom = process_images(dir_anom, 'temp_anom', bound_top, bound_left, bound_bottom, bound_right)
# df_anom['Date'] = pd.to_datetime(df_anom['Date']) - MonthBegin(1)

df_neo = process_neo(dir_neo, bound_top, bound_left, bound_bottom, bound_right)
df_sat = process_sat(dir_calipso)

df_coral = process_images(dir_coral, 'coral_algae', bound_top, bound_left, bound_bottom, bound_right, True)
df_seagrass = process_images("{}/{}".format(dir_geo, 'seagrass'), 'seagrass',
                             bound_top, bound_left, bound_bottom, bound_right, False)
df_rock = process_images("{}/{}".format(dir_geo, 'rock'), 'rock',
                         bound_top, bound_left, bound_bottom, bound_right, False)
df_rubble = process_images("{}/{}".format(dir_geo, 'rubble'), 'rubble',
                           bound_top, bound_left, bound_bottom, bound_right, False)
df_sand = process_images("{}/{}".format(dir_geo, 'sand'), 'sand',
                         bound_top, bound_left, bound_bottom, bound_right, False)
df_reef_slope = process_images("{}/{}".format(dir_geo, 'reef_slope'), 'reef_slope',
                               bound_top, bound_left, bound_bottom, bound_right, False)
df_sheltered_reef_slope = process_images("{}/{}".format(dir_geo, 'sheltered_reef_slope'), 'sheltered_reef_slope',
                                         bound_top, bound_left, bound_bottom, bound_right, False)
df_outer_reef_flat = process_images("{}/{}".format(dir_geo, 'outer_reef_flat'), 'outer_reef_flat',
                                    bound_top, bound_left, bound_bottom, bound_right, False)
df_reef_crest = process_images("{}/{}".format(dir_geo, 'reef_crest'), 'reef_crest',
                               bound_top, bound_left, bound_bottom, bound_right, False)
df_inner_reef_flat = process_images("{}/{}".format(dir_geo, 'inner_reef_flat'), 'inner_reef_flat',
                                    bound_top, bound_left, bound_bottom, bound_right, False)
df_terrestrial_reef_flat = process_images("{}/{}".format(dir_geo, 'terrestrial_reef_flat'), 'terrestrial_reef_flat',
                                          bound_top, bound_left, bound_bottom, bound_right, False)
df_plateau = process_images("{}/{}".format(dir_geo, 'plateau'), 'plateau',
                            bound_top, bound_left, bound_bottom, bound_right, False)
df_backreef = process_images("{}/{}".format(dir_geo, 'backreef'), 'backreef',
                             bound_top, bound_left, bound_bottom, bound_right, False)
df_shallow_lagoon = process_images("{}/{}".format(dir_geo, 'shallow_lagoon'), 'shallow_lagoon',
                                   bound_top, bound_left, bound_bottom, bound_right, False)
df_deep_lagoon = process_images("{}/{}".format(dir_geo, 'deep_lagoon'), 'deep_lagoon',
                                bound_top, bound_left, bound_bottom, bound_right, False)

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


gdf_anom = create_gdf(df_anom)
gdf_coral = create_gdf(df_coral)
gdf_neo = create_gdf(df_neo)
gdf_sat = create_gdf(df_sat)

gdf_seagrass = create_gdf(df_seagrass)
gdf_rock = create_gdf(df_rock)
gdf_rubble = create_gdf(df_rubble)
gdf_sand = create_gdf(df_sand)
gdf_reef_slope = create_gdf(df_reef_slope)
gdf_sheltered_reef_slope = create_gdf(df_sheltered_reef_slope)
gdf_outer_reef_flat = create_gdf(df_outer_reef_flat)
gdf_reef_crest = create_gdf(df_reef_crest)
gdf_inner_reef_flat = create_gdf(df_inner_reef_flat)
gdf_terrestrial_reef_flat = create_gdf(df_terrestrial_reef_flat)
gdf_plateau = create_gdf(df_plateau)
gdf_backreef = create_gdf(df_backreef)
gdf_shallow_lagoon = create_gdf(df_shallow_lagoon)
gdf_deep_lagoon = create_gdf(df_deep_lagoon)


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


all_match = ckdnearest(gdf_coral, gdf_sat)
# TODO: make a dataframe for each month


gdf_list = [gdf_anom, gdf_neo, gdf_seagrass, gdf_rock, gdf_rubble, gdf_sand, gdf_reef_slope,
            gdf_sheltered_reef_slope, gdf_outer_reef_flat, gdf_reef_crest, gdf_inner_reef_flat,
            gdf_terrestrial_reef_flat, gdf_plateau, gdf_backreef, gdf_shallow_lagoon, gdf_deep_lagoon]

for gdf in gdf_list:
    all_match = ckdnearest(all_match, gdf)

all_match.to_csv('data_fusion.csv')

####### Spatial Grid Test
# https://james-brennan.github.io/posts/fast_gridding_geopandas/
xmin, ymin, xmax, ymax = gdf_coral.total_bounds
print(xmin, ymin, xmax, ymax)
# 24.212102963486124 -82.882919311523 26.405982971191 -79.85443162271748
n_cells = 30
cell_size = (xmax - xmin) / n_cells
# projection of the grid
crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"

# create the cells in a loop
grid_cells = []
for x0 in np.arange(xmin, xmax + cell_size, cell_size):
    for y0 in np.arange(ymin, ymax + cell_size, cell_size):
        # bounds
        x1 = x0 - cell_size
        y1 = y0 + cell_size
        grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))
cell = geopandas.GeoDataFrame(grid_cells, columns=['geometry'],
                              crs=crs)

ax = gdf_coral.plot(markersize=.1, figsize=(12, 8), column='coral_algae')
plt.autoscale(False)
cell.plot(ax=ax, facecolor="none", edgecolor='grey')
ax.axis("off")





############
# month-year data matching
############
import datetime

month = 7
startDate = datetime.date(2019, month, 1)
dateRange = pd.date_range(startDate, periods=12, freq='1M')

ts_match = None

gdf_list = [gdf_anom, gdf_neo, gdf_seagrass, gdf_rock, gdf_rubble, gdf_sand, gdf_reef_slope,
            gdf_sheltered_reef_slope, gdf_outer_reef_flat, gdf_reef_crest, gdf_inner_reef_flat,
            gdf_terrestrial_reef_flat, gdf_plateau, gdf_backreef, gdf_shallow_lagoon, gdf_deep_lagoon]

ts_all = None

for x in dateRange:
    sat_sub = gdf_sat[(gdf_sat.Date.dt.month == x.month) & (gdf_sat.Date.dt.year == x.year)]
    anom_sub = gdf_anom[(gdf_anom.Date.dt.month == x.month) & (gdf_anom.Date.dt.year == x.year)]
    neo_sub = gdf_neo[(gdf_neo.Date.dt.month == x.month) & (gdf_neo.Date.dt.year == x.year)]

    gdf_list = [anom_sub, neo_sub, gdf_seagrass, gdf_rock, gdf_rubble, gdf_sand, gdf_reef_slope,
                gdf_sheltered_reef_slope, gdf_outer_reef_flat, gdf_reef_crest, gdf_inner_reef_flat,
                gdf_terrestrial_reef_flat, gdf_plateau, gdf_backreef, gdf_shallow_lagoon, gdf_deep_lagoon]
    # sub_match = ckdnearest(sat_sub, anom_sub)
    ts_match = ckdnearest(sat_sub, gdf_coral)
    for gdf in gdf_list:
        ts_match = ckdnearest(ts_match, gdf)
    if ts_all is None:
        ts_all = ts_match.copy()
    else:
        ts_all = ts_all.append(ts_match)
