from collections import OrderedDict
import copy
import numpy as np
import random

from .geometry.box import Box
from .geometry.vector import Vec3d
from .tensor import TensorData
from . import utils


class OutOfRangeError(Exception):
    def __init__(self):
        super(OutOfRangeError, self).__init__()


class NoSpecError(Exception):
    def __init__(self):
        super(NoSpecError, self).__init__()


class Dataset(object):
    """Dataset for volumetric data.

    Args:
        spec (dictionary): mapping key to tensor's shape.

    Attributes:
        _data (dictionary): mapping key to TensorData.
    """
    def __init__(self, spec=None):
        self._spec = spec
        self._data = dict()
        self._locs = None

    def __call__(self, spec=None):
        return self.random_sample(spec=spec)

    def add_data(self, key, data, offset=(0,0,0)):
        self._data[key] = TensorData(data, offset=offset)

    def add_mask(self, key, data, offset=(0,0,0), loc=False):
        self.add_data(key, data, offset=offset)
        if loc:
            self._locs = dict()
            self._locs['data'] = np.flatnonzero(data)
            self._locs['dims'] = data.shape
            self._locs['offset'] = Vec3d(offset)

    def get_patch(self, key, pos, dim):
        """Extract a patch from the data tagged with `key`."""
        assert key in self._data
        assert len(pos)==3 and len(dim)==3
        return self._data[key].get_patch(pos, dim)

    def get_sample(self, pos, spec=None):
        """Extract a sample centered on pos."""
        spec = self._validate(spec)
        sample = dict()
        for key, dim in spec.items():
            patch = self.get_patch(key, pos, dim[-3:])
            if patch is None:
                raise OutOfRangeError()
            sample[key] = patch
        return utils.sort(sample)

    def random_sample(self, spec=None):
        """Extract a random sample."""
        spec = self._validate(spec)
        try:
            pos = self._random_location(spec)
            ret = self.get_sample(pos, spec)
        except:
            raise OutOfRangeError()
        return ret

    def num_samples(self, spec=None):
        spec = self._validate(spec)
        valid = self._valid_range(spec)
        return np.prod(valid.size())

    ####################################################################
    ## Private Helper Methods.
    ####################################################################

    def _validate(self, spec):
        if spec is None:
            if self._spec is None:
                raise NoSpecError()
            spec = self._spec
        for k in spec:
            assert k in self._data
        return spec

    def _random_location(self, spec):
        """Return a random valid location."""
        valid = self._valid_range(spec)
        if self._locs is None:
            s = valid.size()
            x = np.random.randint(0, s[-1])
            y = np.random.randint(0, s[-2])
            z = np.random.randint(0, s[-3])
            # Global coordinate system.
            loc = Vec3d(z,y,x) + valid.min()
        else:
            while True:
                idx = np.random.choice(self._locs['data'], 1)
                loc = np.unravel_index(idx[0], self._locs['dims'])
                # Global coordinate system.
                loc = Vec3d(loc[-3:]) + self._locs['offset']
                if valid.contains(loc):
                    break
        # DEBUG(kisuk):
        # print('loc = {}'.format(loc))
        return loc

    def _valid_range(self, spec):
        """Compute the valid range, which is intersection of the valid range
        of each TensorData.
        """
        valid = None
        for key, dim in spec.items():
            assert key in self._data
            v = self._data[key].valid_range(dim[-3:])
            valid = v if valid is None else valid.intersect(v)
        return valid
