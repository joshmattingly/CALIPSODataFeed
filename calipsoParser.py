import re

import pandas as pd
import numpy as np
import h5py

import os

from sqlalchemy import create_engine
from getpass import getpass


def process_sat(root):
    dirListing = os.listdir(root)
    for file in dirListing:
        df = None
        print('{}'.format(file))
        if "CAL_LID_L1-Standard-V4-10.2020-06-29T04-38-26ZD_Subset.h5" in file:
            year, month, day = re.findall(r'\.(\d{4})\-(\d{2})\-(\d{2})', file)[0]
            f = h5py.File('{}{}'.format(root, file), 'r')

            if df is None:
                df = pd.DataFrame({'Long': np.array(f['Longitude']).flatten(),
                                   'Lat': np.array(f['Latitude']).flatten(),
                                   'Date': pd.to_datetime(np.array(f['Profile_UTC_Time']).flatten(), format='%y%m%d'),
                                   'Land_Water_Mask': np.array(f['Land_Water_Mask']).flatten()
                                   })
                df_back = pd.DataFrame(np.array(f['Perpendicular_Attenuated_Backscatter_532']))
                # df = pd.concat([df, df_back.iloc[:, -5:]], axis=1)
                df = pd.concat([df, df_back], axis=1)
            else:
                df_temp = pd.DataFrame({'Long': np.array(f['Longitude']).flatten(),
                                        'Lat': np.array(f['Latitude']).flatten(),
                                        'Date': pd.to_datetime(np.array(f['Profile_UTC_Time']).flatten(), format='%y%m%d'),
                                        'Land_Water_Mask': np.array(f['Land_Water_Mask']).flatten()
                                        })
                df_back = pd.DataFrame(np.array(f['Perpendicular_Attenuated_Backscatter_532']))
                # df_combined = pd.concat([df_temp, df_back.iloc[:, -5:]], axis=1)
                df_combined = pd.concat([df_temp, df_back], axis=1)
                df = pd.concat([df, df_combined])
            hostname = "localhost"
            dbname = "coral_data"
            # uname = "root"
            # pwd = getpass()
            engine = create_engine('postgresql://jmattingly31@localhost:5432/coral_data')
            df.to_sql('calipso_data_gbr', engine, index=False, if_exists='replace')
    # return df


if __name__ == '__main__':
    calipso_folder = './2021305125749_62849/'
    process_sat(calipso_folder)
    # df = process_sat(calipso_folder)
    # df.sort_values('Date', inplace=True)

    hostname = "localhost"
    dbname = "coral_data"
    # uname = "root"
    # pwd = getpass()
    engine = create_engine('postgresql://jmattingly31@localhost:5432/coral_data')
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
