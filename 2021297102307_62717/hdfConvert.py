# HDF4 to HDF5 Batch Converter
# Requires H4H5Tools from https://portal.hdfgroup.org/display/support/h4h5tools+2.2.5
# Compile source and place h4toh5convert executable in same folder as downloaded hdf4 files.

import os

dirListing = os.listdir('./')

for file in dirListing:
    if ".hdf" in file:
        print("Processing {}".format(file))
        os.system('./h4toh5convert {}'.format(file))
