#!/usr/bin/env python
__doc__ = """

Utility functions.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

import numpy as np


def check_volume(data):
    """Ensure that data is a numpy 3D array."""
    assert isinstance(data, np.ndarray)

    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
    elif data.ndim == 3:
        pass
    elif data.ndim == 4:
        assert data.shape[0]==1
        data = np.squeeze(data, axis=0)
    else:
        raise RuntimeError("data must be a numpy 3D array")

    assert data.ndim==3
    return data


def check_tensor(data):
    """Ensure that data is a numpy 4D array."""
    assert isinstance(data, np.ndarray)

    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
        data = np.expand_dims(data, axis=0)
    elif data.ndim == 3:
        data = np.expand_dims(data, axis=0)
    elif data.ndim == 4:
        pass
    else:
        raise RuntimeError("data must be a numpy 4D array")

    assert data.ndim==4
    return data
