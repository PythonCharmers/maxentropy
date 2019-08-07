import warnings

from .utils import *


msg = ("This module was deprecated in version 0.3 in favor of the "
       "utils module. This module will be removed in 0.4.")

warnings.warn(msg, DeprecationWarning)


__all__ = ['feature_sampler',
           'dictsample',
           'dictsampler',
           'auxiliary_sampler_scipy',
           'evaluate_feature_matrix',
           'innerprod',
           'innerprodtranspose',
           'DivergenceError']



