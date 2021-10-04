import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re


def process_images(dir, column, top, left, bottom, right, coral=False):
    dir_listing = os.listdir(dir)
    df = None
    for file in dir_listing:
        if ".png" in file:
            print("Processing {}".format(file))
            datestamp = re.findall("(\d{2}\D{3}\d{4})", file)[0]
            img = cv2.imread('{}/{}'.format(dir, file))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray[gray == 19] = 0
            # if detecting coral/fungi, flag sets response to boolean
            if coral:
                gray[gray > 0] = 1
            thresh = cv2.threshold(gray, 100, 200, cv2.THRESH_TOZERO_INV)[1]
            border = np.where(thresh == 0)
            top_thickness = 5
            bottom_thickness = 6
            top_left = (np.min(border[0])+top_thickness, np.min(border[1])+top_thickness)
            bottom_right = (np.max(border[0])-bottom_thickness, np.max(border[1])-bottom_thickness)
            cropped = gray[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
            y, x = cropped.shape
            long_increment = (bottom - top) / y
            lat_increment = (right - left) / x
            lat_range = list(np.arange(left, right, lat_increment))
            long_range = list(np.arange(top, bottom, long_increment))
            if df is None:
                df = pd.DataFrame(cropped, index=long_range, columns=lat_range)
                df = pd.melt(df.reset_index(), id_vars='index')
                df.columns = ['Long', 'Lat', column]
                df['Date'] = pd.to_datetime(datestamp, format="%d%b%Y")
            else:
                while cropped.shape[1] < len(lat_range):
                    lat_range.pop()
                df_temp = pd.DataFrame(cropped, index=long_range, columns=lat_range)
                df_temp = pd.melt(df_temp.reset_index(), id_vars='index')
                df_temp.columns = ['Long', 'Lat', column]
                df_temp['Date'] = pd.to_datetime(datestamp, format="%d%b%Y")
                df = pd.concat([df, df_temp])
    return df


if __name__ == "__main__":
    # test points from CALIPSO_ACA_Subsetter
    dir_anom = '/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/temp_anom'
    dir_coral = '/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/coral_snapshot'
    top = 26.405982971191
    left = -82.882919311523
    right = -79.850692749023
    bottom = 24.208717346191

    df_anom = process_images(dir_anom, 'temp_anom', top, left, bottom, right, False)
    df_coral = process_images(dir_coral, 'coral_fungi', top, left, bottom, right, True)