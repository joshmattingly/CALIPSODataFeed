import urllib
import os
from matplotlib import pyplot as plt

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, shape
import shapely.speedups

from pyspark.sql.functions import *
from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType,DecimalType
from pyspark.sql.functions import lit, pandas_udf, PandasUDFType


gdf_benth = gpd.read_file('/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/'
                                'Benthic-Map/benthic.shp')

gdf_geo = gpd.read_file('/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/'
                              'Geomorphic-Map/geomorphic.shp')

shapely.speedups.enable()
gdf_benth['geometry'] = gdf_benth['geometry'].centroid
gdf_geo['geometry'] = gdf_geo['geometry'].centroid
