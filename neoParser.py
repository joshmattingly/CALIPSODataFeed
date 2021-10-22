import pandas as pd
import numpy as np
import os
import re
import netCDF4 as nc
from getpass import getpass

from sqlalchemy import create_engine

hostname= "localhost"
dbname= "coral_data"
uname = "root"
pwd = getpass()
engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=hostname, db=dbname, user=uname, pw=pwd))

import pymysql


def process_neo(root, top, left, bottom, right):
    # dir_listing = os.listdir(dir)
    for subdir, dirs, files in os.walk(root):
        print("Processing: {}".format(dirs))
        print("Processing: {}".format(subdir))
        for file in files:
            if ".nc" in file:
                df = None
                print("Processing {}".format(file))
                ds = nc.Dataset('{}/{}'.format(subdir, file))
                year = datestamp = ds.time_coverage_end[:4]
                lat_range = list(ds['lat'][:])
                long_range = list(ds['lon'][:])
                if df is None:
                    datestamp = ds.time_coverage_end[:10]
                    df = pd.DataFrame(ds['chlor_a'][:].data, index=lat_range, columns=long_range)
                    df = pd.melt(df.reset_index(), id_vars='index')
                    df.columns = ['Lat', 'Long', 'chlorophyll']
                    df['Date'] = pd.to_datetime(datestamp, format="%Y-%m-%d")
    #                df = df[df['Long'].between(bound_left, bound_right)]
    #                df = df[df['Lat'].between(bound_bottom, bound_top)]
                    df = df[df.chlorophyll > -32767.0]
                    df = df[(df.Long <= right) & (df.Long >= left)]
                    df = df[(df.Lat >= bottom) & (df.Lat <= top)]
                else:
                    datestamp = ds.time_coverage_end[:10]
                    df_temp = pd.DataFrame(ds['chlor_a'][:].data, index=lat_range, columns=long_range)
                    df_temp = pd.melt(df_temp.reset_index(), id_vars='index')
                    df_temp.columns = ['Lat', 'Long', 'chlorophyll']
                    df_temp['Date'] = pd.to_datetime(datestamp, format="%Y-%m-%d")
    #                 df_temp = df_temp[df_temp['Long'].between(bound_left, bound_right)]
    #                df_temp = df_temp[df_temp['Lat'].between(bound_bottom, bound_top)]
                    df = pd.concat([df, df_temp[df_temp.chlorophyll > -32767.0]])
                    df.reset_index()
                # df.to_csv('neo_data_{}.csv'.format(year))
                df.to_sql('neo_data', engine, index=False, if_exists='append')
    # return df.reset_index()


if __name__ == "__main__":
    bound_top = 26.405982971191
    bound_left = -82.882919311523
    bound_right = -79.850692749023
    bound_bottom = 24.208717346191

    root = 'neo_data/'
    process_neo(root, bound_top, bound_left, bound_bottom, bound_right)
    # df_neo.to_csv('neo_data_2014_2020.csv')
