import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('/Users/josh/Google Drive/Georgia Tech Notes/Capstone/data/aca_legend.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_TOZERO_INV)[1]
plt.imshow(thresh)
# border = np.where(thresh == 0)
