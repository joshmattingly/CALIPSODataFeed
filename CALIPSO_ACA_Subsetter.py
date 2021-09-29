import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from skimage import io, color


# coordinates based on CALIPSO area selector
top = 26.405982971191
left = -82.882919311523
right = -79.850692749023
bottom = 24.208717346191

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
