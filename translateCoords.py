import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from skimage import io, color


# https://stackoverflow.com/questions/14034455/translating-lat-long-to-actual-screen-x-y-coordinates-on-a-equirectangular-map

# test
mapWidth = 8640
mapHeight = 4320

lat_min = -180
lng_min = -90

lng_range = 180
lat_range = 360


def getxy(lng, lat, mapWidth = 8640, mapHeight = 4320):
    screen_y = (((lng - lng_min) * mapHeight) + lng_range)
    screen_x = (((lat - lat_min) * mapWidth) + lat_range)
    return round(screen_y), round(screen_x)


def getlonglat(x, y, mapWidth = 8640, mapHeight = 4320):
    new_lng = ((x * lng_range) / mapHeight) + lng_min
    new_lat = ((x * lat_range) / mapWidth) + lat_min

    return new_lat, new_lng


def getmapslice(longs, lats):
    topLeft = getxy(longs[0], lats[0])
    topRight = getxy(longs[0], lats[1])
    bottomLeft = getxy(longs[1], lats[0])
    bottomRight = getxy(longs[1], lats[1])
    return [topLeft, topRight, bottomLeft, bottomRight]
