#!/usr/bin/env python
__doc__ = """

Dataset classes.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

from collections import OrderedDict
import copy
import numpy as np

from .geometry.box import Box
from .geometry.vector import Vec3d
from .tensor import TensorData
from . import utils


class VolumetricDataset(object):
    """
    Dataset for volumetric data.

    Attributes:
        _data: Dictionary mapping key to TensorData containing
                4D volumetric data. (e.g. EM image stack, segmentation, etc.)
    """

    def __init__(self, **kwargs):
        # Initialize attributes.
        self._reset()

    def add_data(self, key, data, offset=(0,0,0)):
        """
        Add a volumetric data to dataset.

        Args:
            key (string): data tag.
            data (ndarray): volumetric data of size (z,y,x) or (c,z,y,x).
            offset (tuple of three ints, optional): offset from the origin.
        """
        self._data[key] = TensorData(data, offset=offset)

    def get_sample(self, pos, spec):
        """Extract a sample centered on pos."""
        sample = dict()
        for key, dim in spec.items():
            assert key in self._data
            patch = self._data[key].get_patch(pos, dim[-3:])
            if patch is None:
                raise
            else:
                sample[key] = patch
        return OrderedDict(sorted(sample.items(), key=lambda x: x[0]))

    def random_sample(self, spec):
        """Extract a random sample."""
        try:
            # Pick a random sample.
            pos = self._random_location(spec)
            ret = self.get_sample(pos, spec)
        except:
            raise
        return ret

    ####################################################################
    ## Getters and setters.
    ####################################################################

    def num_sample(self, spec):
        raise NotImplementedError

    ####################################################################
    ## Private Helper Methods.
    ####################################################################

    def _reset(self):
        """Reset all attributes."""
        self._data = dict()

    def _random_location(self, spec):
        """Return one of the valid locations randomly."""
        valid = self._valid_range(spec)  # Valid range.
        fails = 0
        s = valid.size()
        z = np.random.randint(0, s[0])
        y = np.random.randint(0, s[1])
        x = np.random.randint(0, s[2])
        # Global coordinate system.
        loc =  Vec3d(z,y,x) + valid.min()
        # DEBUG(kisuk)
        # print("loc = {}, fails = {}".format(loc, fails))
        return loc

    def _valid_range(self, spec):
        """
        Compute the valid range.
        Compute the intersection of the valid range of each TensorData.
        """
        ret = None
        for key, dim in spec.items():
            assert key in self._data
            v = self._data[key].valid_range(dim[-3:])
            ret = v if ret is None else ret.intersect(v)
        return ret
