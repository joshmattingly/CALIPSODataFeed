import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from skimage import io, color


# https://stackoverflow.com/questions/14034455/translating-lat-long-to-actual-screen-x-y-coordinates-on-a-equirectangular-map

# test
mapWidth = 3600
mapHeight = 1800

lat_min = -180
lng_min = -90

lng_range = 180
lat_range = 360


def getxy(lng, lat, mapWidth = 36000, mapHeight = 18000):
    screen_y = (((lng - lng_min) * mapHeight) + lng_range)
    screen_x = (((lat - lat_min) * mapWidth) + lat_range)
    return round(screen_y), round(screen_x)


def getlonglat(x, y, mapWidth = 36000, mapHeight = 18000):
    new_lng = ((x * lng_range) / mapHeight) + lng_min
    new_lat = ((x * lat_range) / mapWidth) + lat_min

    return new_lat, new_lng


def getmapslice(longs, lats):
    topLeft = getxy(longs[0], lats[0])
    topRight = getxy(longs[0], lats[1])
    bottomLeft = getxy(longs[1], lats[0])
    bottomRight = getxy(longs[1], lats[1])
    return [topLeft, topRight, bottomLeft, bottomRight]


x, y = getxy(41.9495103, -87.7332445)
# Out[14]: (923, 841)
lng, lat = getlonglat(x, y)
# Out[15]: (41.9, -87.7)

img = Image\
    .open('/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/NEO/MY1DMM_CHLORA_2021-08-01_rgb_3600x1800.TIFF')
img = img.resize((36000, 18000))
img_array = np.asarray(img)
# plt.imshow(img_array)
mapHeight, mapWidth = img_array.shape

img_array.shape

# South Florida Test
longs = [26.466407775879, 24.368019104004]
lats = [-82.409477233887, -79.629936218262]

coords = getmapslice(longs, lats, mapWidth, mapHeight)
# Out[40]: [(976, 635), (1004, 635), (976, 656), (1004, 656)]

map_window = img_array[coords[0][1]:coords[2][1], coords[0][0]:coords[1][0]]

plt.imshow(map_window)
pd.DataFrame(map_window).to_csv('test_zoom.csv')

