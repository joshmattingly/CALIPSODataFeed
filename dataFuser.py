import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from acaParser import process_images
from neoParser import process_neo
from calipsoParser import process_sat


dir_anom = '/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/temp_anom'
dir_coral = '/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/coral_snapshot'
dir_neo = '/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/neo_data'
dir_calipso = '/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/2021272175244_62409/'

bound_top = 26.405982971191
bound_left = -82.882919311523
bound_right = -79.850692749023
bound_bottom = 24.208717346191

df_anom = process_images(dir_anom, 'temp_anom', bound_top, bound_left, bound_bottom, bound_right, False)
df_coral = process_images(dir_coral, 'coral_fungi', bound_top, bound_left, bound_bottom, bound_right, True)
df_neo = process_neo(dir_neo, bound_top, bound_left, bound_bottom, bound_right)
df_sat = process_sat(dir_calipso)
