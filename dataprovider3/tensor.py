#!/usr/bin/env python
__doc__ = """

Read-only/writable TensorData classes.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

import math
import numpy as np
import time

from .geometry.box import Box, centered_box
from .geometry.vector import Vec3d
from . import utils


class TensorData(object):
    """
    Read-only tensor data.

    The 1st dimension is regarded as channels, and arbitrary access
    in this dimension is not allowed. Threfore, every data access should be
    made through a 3D vector, not 4D.

    Attributes:
        _data:   4D numpy array. (channel, z, y, x)
        _dim:    Dimension of each channel.
        _offset: Coordinate offset from the origin.
        _bbox:   Bounding box.
    """

    def __init__(self, data, offset=(0,0,0)):
        """Initialize a TensorData object."""
        self._data   = utils.check_tensor(data)
        self._dim    = Vec3d(self._data.shape[-3:])
        self._offset = Vec3d(offset)
        # Set bounding box.
        self._bbox = Box((0,0,0), self._dim)
        self._bbox.translate(self._offset)

    def get_patch(self, pos, dim):
        """Extract a patch of size `dim` centered on `pos`."""
        assert len(pos)==3 and len(dim)==3
        patch = None
        # Is the patch contained within the bounding box?
        box = centered_box(pos, dim)
        if self._bbox.contains(box):
            box.translate(-self._offset)  # Local coordinate system.
            vmin  = box.min()
            vmax  = box.max()
            patch = np.copy(self._data[:,vmin[0]:vmax[0],
                                         vmin[1]:vmax[1],
                                         vmin[2]:vmax[2]])
        return patch

    def valid_range(self, dim):
        """Get a valid range for extracting patches of size `dim`."""
        assert len(dim)==3
        dim  = Vec3d(dim)
        top  = dim // 2             # Top margin.
        btm  = dim - top - (1,1,1)  # Bottom margin.
        vmin = self._offset + top
        vmax = self._offset + self._dim - btm
        return Box(vmin, vmax)

    ####################################################################
    ## Public methods for accessing attributes.
    ####################################################################

    def data(self):
        return self._data

    def shape(self):
        """Return data shape (c,z,y,x)."""
        return self._data.shape

    def dim(self):
        """Return channel shape (z,y,x)."""
        return Vec3d(self._dim)

    def offset(self):
        return Vec3d(self._offset)

    def bbox(self):
        return Box(self._bbox)

    ####################################################################
    ## Private helper methods.
    ####################################################################

    def __str__( self ):
        return "<TensorData>\nshape: %s\ndim: %s\noffset: %s\n" % \
               (self.shape(), self._dim, self._offset)


########################################################################
## Unit Testing
########################################################################
if __name__ == "__main__":

    import unittest

    ####################################################################
    class UnitTestTensorData(unittest.TestCase):

        def setup(self):
            pass

        def testCreation(self):
            data = np.zeros((4,4,4,4))
            T = TensorData(data, (1,1,1))
            self.assertTrue(T.shape()==(4,4,4,4))
            self.assertTrue(T.offset()==(1,1,1))
            bb = T.bbox()
            self.assertTrue(bb==Box((1,1,1),(5,5,5)))

        def testGetPatch(self):
            # (4,4,4) random 3D araray
            data = np.random.rand(4,4,4)
            dim = (3,3,3)
            T = TensorData(data)
            p = T.get_patch((2,2,2), dim)
            self.assertTrue(np.array_equal(data[1:,1:,1:], p[0,...]))
            dim = (2,2,2)
            p = T.get_patch((2,2,2), dim)
            self.assertTrue(np.array_equal(data[1:3,1:3,1:3], p[0,...]))
            p = T.get_patch((3,3,3), (3,3,3))
            self.assertEqual(p, None)

    ####################################################################
    unittest.main()

    ####################################################################
