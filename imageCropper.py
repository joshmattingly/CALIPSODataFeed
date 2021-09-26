import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
from cv2 import matchTemplate as cv2m

img = cv2.imread('/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/stripped_down.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 75, 150)


# TODO: find top left
# finding the top left corner will let the cropper know where the lat-long anchor begins.
corner = np.where(gray == 147)
# gray[34:, 41:]
img_cropped = gray[corner[0][0]+1:, corner[1][0]+1:]
