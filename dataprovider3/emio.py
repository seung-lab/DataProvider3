#!/usr/bin/env python
__doc__ = """

I/O for EM images.

Adopted the code by Jingpeng Wu <jingpeng.wu@gmail.com>

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

import h5py
import numpy as np
import os


def imread(fname):
    _, ext = os.path.splitext(fname)
    if ext == '.h5' or ext == '.hdf5':
        with h5py.File(fname, 'r') as f:
            data = np.asarray(f['/main'])
    else:
        raise RuntimeError("only hdf5 format is supported")
    return data


def imsave(data, fname):
    _, ext = os.path.splitext(fname)
    if ext == '.h5' or ext == '.hdf5':
        with h5py.File(fname, 'w') as f:
            f.create_dataset('/main', data=data)
    else:
        raise RuntimeError("only hdf5 format is supported")
