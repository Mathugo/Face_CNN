import numpy as np
import os
import h5py
from PIL import Image
import io
"""
    Saves compressed, resized images as HDF5 datsets
    Returns
        data.h5, where each dataset is an image or class label
        e.g. X23,y23 = image and corresponding class label
"""

class proc_h5:
    def __init__(self):
       pass
    def print_h5(self, hdf5):
        f = h5py.File(hdf5)
        dset_read = f.get('binary_data')
        dset_read_np = np.array(dset_read)
        img_res = Image.open(io.BytesIO(dset_read_np))
        img_res.show()
        f.close()

    def jpg_to_h5(self, filename, hdf5):
        with open(filename, 'rb') as img_f:
            image_file = img_f.read()

        #print(type(image_file)) # <class 'bytes'>  
        img_np_array = np.asarray(image_file)
        #print(type(img_np_array)) # <class 'numpy.ndarray'>
        # store to hdf5 file
        f = h5py.File(hdf5)
        dset = f.create_dataset('binary_data', data=img_np_array)
        f.close()
