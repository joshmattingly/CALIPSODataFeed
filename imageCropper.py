import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


img = cv2.imread('/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/ACA_DHW/DHW21Oct2019.png')
# img = cv2.imread('/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/coral_snapshot.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray[gray == 19] = 0
gray[gray > 0] = 1
# edges = cv2.Canny(gray, 75, 150)

# find the bounding box
# https://stackoverflow.com/questions/55169645/square-detection-in-image
thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_TOZERO_INV)[1]
plt.imshow(thresh)
border = np.where(thresh == 0)

# find the top left and bottom right corners of the bounding box

# shift indicies to start one pixel in from box
top_thickness = 5
bottom_thickness = 6
top_left = (np.min(border[0])+top_thickness, np.min(border[1])+top_thickness)
bottom_right = (np.max(border[0])-bottom_thickness, np.max(border[1])-bottom_thickness)

cropped = gray[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
plt.imshow(cropped)
'''
# TODO: find top left
# finding the top left corner will let the cropper know where the lat-long anchor begins.
corner = np.where(gray == 147)
# gray[34:, 41:]
img_cropped = gray[corner[0][0]+1:, corner[1][0]+1:]
'''

y, x = cropped.shape

# test points from CALIPSO_ACA_Subsetter
top = 26.405982971191
left = -82.882919311523
right = -79.850692749023
bottom = 24.208717346191

long_increment = (bottom - top) / y
lat_increment = (right - left) / x

lat_range = list(np.arange(left, right, lat_increment))
long_range = list(np.arange(top, bottom, long_increment))

df = pd.DataFrame(cropped, index=long_range, columns=lat_range)
df = pd.melt(df.reset_index(), id_vars='index')
df.columns = ['Long', 'Lat', 'Bleaching']
# df.columns = ['Long', 'Lat', 'Coral/Fungi']
