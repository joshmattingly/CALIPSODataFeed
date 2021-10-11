import pandas as pd
import numpy as np
import os
import re
import netCDF4 as nc


def process_neo(dir, bound_top, bound_left, bound_bottom, bound_right,
                top=90.0, left=-180.0, bottom=-90.0, right=180.0):
    dir_listing = os.listdir(dir)
    df = None
    for file in dir_listing:

        if ".nc" in file:
            print("Processing {}".format(file))
            ds = nc.Dataset('{}/{}'.format(dir, file))
            datestamp = re.findall("(\d{2}\D{3}\d{4})", file)[0]

            lat_range = list(ds['lat'][:])
            long_range = list(ds['lon'][:])
            if df is None:
                df = pd.DataFrame(ds['chlor_a'][:].data, index=lat_range, columns=long_range)
                df = pd.melt(df.reset_index(), id_vars='index')
                df.columns = ['Lat', 'Long', 'chlorophyll']
                df['Date'] = pd.to_datetime(datestamp, format="%d%b%Y")
                df = df[df['Long'].between(bound_left, bound_right)]
                df = df[df['Lat'].between(bound_bottom, bound_top)]
            else:
                df_temp = pd.DataFrame(ds['chlor_a'][:].data, index=lat_range, columns=long_range)
                df_temp = pd.melt(df_temp.reset_index(), id_vars='index')
                df_temp.columns = ['Lat', 'Long', 'chlorophyll']
                df_temp['Date'] = pd.to_datetime(datestamp, format="%d%b%Y")
                df_temp = df_temp[df_temp['Long'].between(bound_left, bound_right)]
                df_temp = df_temp[df_temp['Lat'].between(bound_bottom, bound_top)]
                df = pd.concat([df, df_temp])
    return df.reset_index()


if __name__ == "__main__":
    dir_neo = '/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/neo_data/test'
    bound_top = 26.405982971191
    bound_left = -82.882919311523
    bound_right = -79.850692749023
    bound_bottom = 24.208717346191

    df_neo = process_neo(dir_neo, bound_top, bound_left, bound_bottom, bound_right)
