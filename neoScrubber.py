import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from skimage import io, color


# https://stackoverflow.com/questions/14034455/translating-lat-long-to-actual-screen-x-y-coordinates-on-a-equirectangular-map

def getxy(lng, lat):
    screenX = ((lng + 180) * (mapWidth / 360))
    screenY = (((lat * -1) + 90) * (mapHeight / 180))
    return round(screenY), round(screenX)


def getlonglat(x, y):
    lng = (-180 * mapWidth + 360 * x) / mapWidth
    lat = (90 * mapHeight - 180 * y)/mapHeight

    return lat, lng


def getmapslice(longs, lats):
    topLeft = getxy(longs[0], lats[0])
    topRight = getxy(longs[0], lats[1])
    bottomLeft = getxy(longs[1], lats[0])
    bottomRight = getxy(longs[1], lats[1])
    return [topLeft, topRight, bottomLeft, bottomRight]


# test
mapWidth = 3600
mapHeight = 1800

x, y = getxy(41.9495103, -87.7332445)
# Out[14]: (923, 841)
lng, lat = getlonglat(x, y)
# Out[15]: (41.9, -87.7)

img = Image\
    .open('/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/NEO/MY1DMM_CHLORA_2021-08-01_rgb_3600x1800.TIFF')
img_array = np.asarray(img)
plt.imshow(img_array)
mapHeight, mapWidth = img_array.shape

img_array.shape

# South Florida Test
longs = [26.466407775879, 24.368019104004]
lats = [-82.409477233887, -79.629936218262]

coords = getmapslice(longs, lats)
# Out[40]: [(976, 635), (1004, 635), (976, 656), (1004, 656)]

map_window = img_array[coords[0][1]:coords[2][1], coords[0][0]:coords[1][0]]

plt.imshow(map_window)
pd.DataFrame(map_window).to_csv('test_zoom.csv')

