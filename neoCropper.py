import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
import netCDF4 as nc


'''
def process_neo(dir, bound_top, bound_left, bound_bottom, bound_right,
                top=90.0, left=-180.0, bottom=-90.0, right=180.0):
    dir_listing = os.listdir(dir)
    df = None
    for file in dir_listing:
        if ".nc" in file:
            print("Processing {}".format(file))
            datestamp = re.findall("(\d{2}\D{3}\d{4})", file)[0]

            if df is None:
                df = pd.DataFrame(gray, index=long_range, columns=lat_range)
                df = pd.melt(df.reset_index(), id_vars='index')
                df.columns = ['Long', 'Lat', 'chlorophyll']
                df['Date'] = pd.to_datetime(datestamp, format="%d%b%Y")
            else:
                df_temp = pd.DataFrame(gray, index=long_range, columns=lat_range)
                df_temp = pd.melt(df_temp.reset_index(), id_vars='index')
                df_temp.columns = ['Long', 'Lat', 'chlorophyll']
                df_temp['Date'] = pd.to_datetime(datestamp, format="%d%b%Y")
                df = pd.concat([df, df_temp])
    df = df[df['Long'].between(bound_bottom, bound_top)]
    df = df[df['Lat'].between(bound_left, bound_right)]
    return df
'''

if __name__ == "__main__":
    dir_neo = '/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/neo_data'
    bound_top = 26.405982971191
    bound_left = -82.882919311523
    bound_right = -79.850692749023
    bound_bottom = 24.208717346191

    # df_neo = process_neo(dir_neo, bound_top, bound_left, bound_bottom, bound_right)

    ds = nc.Dataset('{}/{}'.format(dir_neo, 'A20212132021243.L3m_MO_CHL_chlor_a_4km.nc'))
    for dim in ds.dimensions.values():
        print(dim)

    for var in ds.variables.values():
        print(var)

    chlor = ds['chlor_a']
