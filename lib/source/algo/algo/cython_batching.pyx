import cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cimport numpy as np
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef cython_flatten(object[int, ndim=1] indices, object[double, ndim=1]  dist_matrix_flat):
  cdef int num = sizeof(indices) // sizeof(int)
  cdef np.ndarray[double, ndim=1] batch = np.zeros(indices.shape[0], dtype='double')

  for i in xrange(num):
    batch[i] = dist_matrix_flat[i]
  return batch