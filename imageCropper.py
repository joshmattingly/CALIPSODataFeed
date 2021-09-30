import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def process_image(dir, top, left, bottom, right):
    dirListing = os.listdir(dir)

    for file in dirListing:
        if ".pngs" in file:
            print("Processing {}".format(file))
            img = cv2.imread('{}/{}'.format(dir, file))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray[gray == 19] = 0
            gray[gray > 0] = 1
            thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_TOZERO_INV)[1]
            plt.imshow(thresh)
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

            df = pd.DataFrame(cropped, index=long_range, columns=lat_range)
            df = pd.melt(df.reset_index(), id_vars='index')
            df.columns = ['Long', 'Lat', 'Bleaching']


if __name__ == "__main__":
    # test points from CALIPSO_ACA_Subsetter
    dir = '/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/2021272175244_62409'
    top = 26.405982971191
    left = -82.882919311523
    right = -79.850692749023
    bottom = 24.208717346191

    process_image(dir, top, left, bottom, right)
