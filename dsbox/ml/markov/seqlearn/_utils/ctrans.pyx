# Copyright Lars Buitinck, Mikhail Korobov 2013.

import cython
import numpy as np

cimport numpy as cnp
cnp.import_array()

# TODO handle second-order transitions (trigrams)
@cython.boundscheck(False)
@cython.wraparound(False)
def count_trans(cnp.ndarray[ndim=1, dtype=cnp.npy_intp] y, n_classes):
    """Count transitions in a target vector.

    Parameters
    ----------
    y : array of integers, shape = n_samples
    n_classes : int
        Number of distinct labels.
    """
    cdef cnp.ndarray[ndim=2, dtype=cnp.npy_intp, mode='c'] trans
    cdef cnp.npy_intp i

    trans = np.zeros((n_classes, n_classes), dtype=np.intp)

    for i in range(y.shape[0] - 1):
        trans[y[i], y[i + 1]] += 1
    return trans
