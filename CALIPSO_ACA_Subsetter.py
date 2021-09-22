import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from skimage import io, color

from neoScrubber import getlonglat, getxy

# coordinates based on CALIPSO area selector
top = 28.482055664063
left = -86.149291992188
right = -64.000854492188
bottom = 15.825805664063

top_left = (top, left)
top_right = (top, right)
bottom_left = (bottom, left)
bottom_right = (bottom, right)

# for copying and pasting into ACA
print(top_left)
print(top_right)
print(bottom_right)
print(bottom_left)
print(top_left)

# import the ACA image and convert to 1D grayscale, casting values to integers
img = Image \
    .open('/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/Screen Shot 2021-09-22 at 7.28.06 AM.png')
img_array = np.asarray(img)
img_gray = np.uint8(color.rgb2gray(color.rgba2rgb(img_array))*255)
plt.imshow(img_gray)

img_gray.shape
# Out[15]: (393, 632)

# based on NEO full map
mapWidth = 3600
mapHeight = 1800

# create empty numpy array based on NEO map dimensions
aca_map = np.zeros((mapHeight, mapWidth))

# use getxy function from neoScrubber.py to figure out where the ACA window should live in full array
homeLocation = getxy(left, top)
