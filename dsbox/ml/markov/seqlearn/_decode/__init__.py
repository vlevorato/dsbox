# Copyright Lars Buitinck 2013.

"""Decoding (inference) algorithms."""

import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})

from .bestfirst import bestfirst
from .viterbi import viterbi

DECODERS = {"bestfirst": bestfirst,
            "viterbi": viterbi}
