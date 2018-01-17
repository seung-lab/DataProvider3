#!/usr/bin/env python
__doc__ = """

Dataset classes.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

from collections import OrderedDict
import copy
import numpy as np
import random

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

    def __init__(self, sample_spec):
        """
        Args:
            sample_spec (dict): default sample spec.
        """
        self._spec = sample_spec
        self._data = dict()

    def add_data(self, key, data, offset=(0,0,0)):
        """Add a volumetric data to dataset.

        Args:
            key (string): data tag.
            data (ndarray): volumetric data of size (z,y,x) or (c,z,y,x).
            offset (tuple of three ints, optional): offset from the origin.
        """
        self._data[key] = TensorData(data, offset=offset)

    def get_sample(self, pos, spec=None):
        """Extract a sample centered on pos."""
        if spec is None:
            spec = self._spec

        sample = dict()
        for key, dim in spec.items():
            patch = self.get_patch(key, pos, dim[-3:])
            if patch is None:
                raise
            else:
                sample[key] = patch

        return OrderedDict(sorted(sample.items(), key=lambda x: x[0]))

    def get_patch(self, key, pos, dim):
        """Extract a patch from the data tagged with `key`."""
        assert key in self._data
        assert len(pos)==3 and len(dim)==3
        return self._data[key].get_patch(pos, dim)

    def random_sample(self, spec=None):
        """Extract a random sample."""
        if spec is None:
            spec = self._spec
        try:
            # Pick a random sample.
            pos = self._random_location(spec)
            ret = self.get_sample(pos, spec)
        except:
            raise
        return ret

    def num_samples(self, spec=None):
        if spec is None:
            spec = self._spec
        valid = self._valid_range(spec)
        return np.prod(valid.size())


    ####################################################################
    ## Private Helper Methods.
    ####################################################################

    def _random_location(self, spec):
        """Return one of the valid locations randomly."""
        valid = self._valid_range(spec)  # Valid range.
        s = valid.size()
        # z = np.random.randint(0, s[0])
        # y = np.random.randint(0, s[1])
        # x = np.random.randint(0, s[2])
        z = random.randint(0, s[0]-1)
        y = random.randint(0, s[1]-1)
        x = random.randint(0, s[2]-1)
        # Global coordinate system.
        loc = Vec3d(z,y,x) + valid.min()
        # DEBUG(kisuk)
        # print("loc = {}".format(loc))
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
