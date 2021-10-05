import pandas as pd
import numpy as np
import h5py

import os


def process_sat(root):
    dirListing = os.listdir(root)
    df = None
    for file in dirListing:
        if ".h5" in file:
            f = h5py.File('{}{}'.format(root, file), 'r')

            if df is None:
                df = pd.DataFrame({'Lat': np.array(f['Latitude']).flatten(),
                                   'Long': np.array(f['Longitude']).flatten(),
                                   'Date': pd.to_datetime(np.array(f['Profile_UTC_Time']).flatten(), format='%y%m%d'),
                                   'Land_Water_Mask': np.array(f['Land_Water_Mask']).flatten()
                                   })
                df_back = pd.DataFrame(np.array(f['Perpendicular_Attenuated_Backscatter_532']))
                df = pd.concat([df, df_back.iloc[:, -5:]], axis=1)
            else:
                df_temp = pd.DataFrame({'Lat': np.array(f['Latitude']).flatten(),
                                        'Long': np.array(f['Longitude']).flatten(),
                                        'Date': pd.to_datetime(np.array(f['Profile_UTC_Time']).flatten(), format='%y%m%d'),
                                        'Land_Water_Mask': np.array(f['Land_Water_Mask']).flatten()
                                        })
                df_back = pd.DataFrame(np.array(f['Perpendicular_Attenuated_Backscatter_532']))
                df_combined = pd.concat([df_temp, df_back.iloc[:, -5:]], axis=1)
                df = pd.concat([df, df_combined])
    return df


if __name__ == '__main__':
    calipso_folder = '/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/2021272175244_62409/'
    df = process_sat(calipso_folder)
