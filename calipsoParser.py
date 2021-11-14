import re

import pandas as pd
import numpy as np
import h5py

import os

from sqlalchemy import create_engine
from getpass import getpass
import geopandas
from scipy.spatial import cKDTree

import warnings
warnings.filterwarnings("ignore")


def create_gdf(df):
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Long, df.Lat),
                                 crs="+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs")
    gdf.drop(['Lat', 'Long'], axis=1, inplace=True)
    return gdf


def get_aim(engine):
    # print("Importing AIM data")
    df_manta = geopandas.read_postgis("SELECT * FROM manta_tow_data", con=engine, geom_col='geom')
    return df_manta


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


def process_sat(root, manta=False):
    dirListing = os.listdir(root)
    hostname = "localhost"
    dbname = "coral_data"
    engine = create_engine('postgresql://jmattingly31@localhost:5432/coral_data')

    # local dev connection
    # engine = create_engine('postgresql://{}@{}:{}/{}'.format("jmattingly31", "127.0.0.1",
    #                                                          local_port, "coral_data"))

    if manta:
        df_manta = get_aim(engine)
        df_manta_dates = pd.DataFrame(pd.to_datetime(df_manta['sample_date'].unique(),
                                                     format='%Y-%m-%d'), columns=['sample_date'])
        df_manta_dates.sort_values('sample_date', inplace=True)
        gdf_manta = create_gdf(df_manta)

    for file in dirListing:
        df = None
        print('{}'.format(file))
        if ".h5" in file:
            year, month, day = re.findall(r'\.(\d{4})\-(\d{2})\-(\d{2})', file)[0]
            f = h5py.File('{}{}'.format(root, file), 'r')

            if df is None:
                df = pd.DataFrame({'Long': np.array(f['Longitude']).flatten().byteswap().newbyteorder(),
                                   'Lat': np.array(f['Latitude']).flatten().byteswap().newbyteorder(),
                                   'Date': pd.to_datetime(np.array(f['Profile_UTC_Time']).flatten()
                                                          .byteswap().newbyteorder(), format='%y%m%d'),
                                   'Land_Water_Mask': np.array(f['Land_Water_Mask']).flatten().byteswap().newbyteorder()
                                   })
                df_back = pd.DataFrame(np.array(f['Perpendicular_Attenuated_Backscatter_532'])
                                       .byteswap().newbyteorder())
                # df = pd.concat([df, df_back.iloc[:, -5:]], axis=1)
                df = pd.concat([df, df_back], axis=1)
            else:
                df_temp = pd.DataFrame({'Long': np.array(f['Longitude']).flatten().byteswap().newbyteorder(),
                                        'Lat': np.array(f['Latitude']).flatten().byteswap().newbyteorder(),
                                        'Date': pd.to_datetime(np.array(f['Profile_UTC_Time']).flatten()
                                                               .byteswap().newbyteorder(), format='%y%m%d'),
                                        'Land_Water_Mask': np.array(f['Land_Water_Mask']).flatten()
                                       .byteswap().newbyteorder()
                                        })
                df_back = pd.DataFrame(np.array(f['Perpendicular_Attenuated_Backscatter_532'])
                                       .byteswap().newbyteorder())
                # df_combined = pd.concat([df_temp, df_back.iloc[:, -5:]], axis=1)
                df_combined = pd.concat([df_temp, df_back], axis=1)
                df = pd.concat([df, df_combined])

            # because of the size of the Great Barrier Reef CALIPSO pull, each file needs to be processed
            # individually before being uploaded POSTGIS
            if manta:
                # convert dataframe to geopandas
                df['Latitude'] = df.Lat
                df['Longitude'] = df.Long
                gdf = create_gdf(df)
                gdf_calipso_manta = pd.merge_asof(gdf, df_manta_dates,
                                                 left_on='Date', right_on='sample_date', direction='nearest')

                # make sure proper column name for nearest neighbor
                # TODO: pass column name into funciton
                try:
                    gdf_calipso_manta.rename(columns={'geometry': 'geom'}, inplace=True)
                except:
                    print('failed to change column name')
                    continue

                gdf_manta_final = None

                for idx, row in df_manta_dates.iterrows():
                    date = row['sample_date']
                    # print(date)
                    try:
                        temp_calipso = gdf_calipso_manta[gdf_calipso_manta.sample_date == date]
                        temp_manta = gdf_manta[gdf_manta.sample_date == date]
                        temp_gdf = ckdnearest(temp_calipso, temp_manta)
                        if gdf_manta_final is None:
                            gdf_manta_final = temp_gdf
                        else:
                            gdf_manta_final = gdf.append(temp_gdf)
                    except:
                        # print("Error processing {}".format(date))
                        continue

                # match manta tow data without temporal alignment.
                # gdf_manta_final = ckdnearest(gdf_calipso_manta, gdf_manta)

                gdf_manta_final = gdf_manta_final[gdf_manta_final.dist <= 1.]
                # gdf_manta_final.drop(['geom', 'geometry'], axis=1, inplace=True)
                gdf_manta_final.rename(columns={'Longitude': 'Long', 'Latitude': 'Lat'}, inplace=True)
                gdf_manta_final.drop(['geom', 'geometry'], axis=1, inplace=True)
                gdf_manta_final.to_sql('calipso_data_gbr', engine, index=False, if_exists='append')
                # return gdf_manta_final
    # return df


if __name__ == '__main__':

    hostname = "localhost"
    dbname = "coral_data"
    # uname = "root"
    # pwd = getpass()
    # engine = create_engine('postgresql://jmattingly31@localhost:5432/coral_data')
    calipso_folder = './2021317092632_63059/'
    # gdf = process_sat(calipso_folder, manta=True)
    process_sat(calipso_folder, manta=True)
    # df = process_sat(calipso_folder, manta=True)
    # df.sort_values('Date', inplace=True)
    # df_neo_dates = pd.read_sql('SELECT a."Date" FROM neo_data a GROUP BY a."Date";', con=engine)
    # df_neo_dates.sort_values('Date', inplace=True)
    # df_neo_dates['NEO_date'] = df_neo_dates['Date']
    # A20203292020336.L3m_8D_CHL_chlor_a_4km.nc
    # df = pd.merge_asof(left=df, right=df_neo_dates, on='Date', direction='forward')

    #df['year'] = df.NEO_date.dt.year.values
    #df['to_date'] = df.apply(lambda row: str(row['NEO_date'].day_of_year).zfill(3), axis=1)
    #df['from_date'] = df.apply(lambda row: str(row['NEO_date'].day_of_year-8).zfill(3), axis=1)
    #df['NEO_file'] = df.apply(lambda row: "A{}{}{}{}.L3m_8D_CHL_chlor_a_4km.nc".format(row['year'],
    #                                                                                   row['from_date'],
    #                                                                                   row['year'],
    #                                                                                   row['to_date']),
    #                          axis=1)
    # from_date = (to_date - 8).zfill(3)
    # df.to_csv('calipso.csv')
    # df.to_sql('calipso_data_gbr', engine, index=False, if_exists='replace')
