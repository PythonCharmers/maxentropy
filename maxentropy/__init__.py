"""
# maxentropy: Routines for fitting maximum entropy models.

Copyright: Ed Schofield, 2003-2019 License: BSD-style (see LICENSE.txt in
main source directory)

Routines for fitting two kinds of models:
- maximum entropy
- minimum KL-divergence

subject to linear expectation constraints.

Contains two classes, one for small models, the other for large models.
Here a 'small' model is one defined on a discrete sample space small
enough to sum over in practice, whereas a 'large' model is on a sample
space that is either continuous (possibly high-dimensional) or discrete
but too large to sum over, requiring Monte Carlo simulation.


## Usage:

Use the .fit() method to fit the maxent model p whose feature
expectations are given by the vector K.

Model expectations are computed either exactly or using Monte Carlo
simulation, depending on the 'func' and 'grad' parameters passed to this
function.

For 'Model' instances, expectations are computed exactly, by summing over
the given sample space.  If the sample space is continuous or too large
to iterate over, use the 'BigModel' class instead.

For 'BigModel' instances, the model expectations are not computed exactly
(by summing or integrating over a sample space) but approximately (by
Monte Carlo simulation).  Simulation is necessary when the sample space
is too large to sum or integrate over in practice, like a continuous
sample space in more than about 4 dimensions or a large discrete space
like all possible sentences in a natural language.

Approximating the expectations by sampling requires an instrumental
distribution that should be close to the model for fast convergence.  The
tails should be fatter than the model.  This instrumental distribution is
specified by calling setsampleFgen() with a user-supplied generator
function that yields a matrix of features of a random sample and its log
pdf values.


## Algorithms:

The algorithm can be 'CG', 'BFGS', 'LBFGSB', 'Powell', or 'Nelder-Mead'.

The CG (conjugate gradients) method is the default; it is quite fast and
requires only linear space in the number of parameters, (not quadratic,
like Newton-based methods).

The BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm is a variable
metric Newton method.  It is perhaps faster than the CG method but
requires O(N^2) instead of O(N) memory, so it is infeasible for more than
about 10^3 parameters.

The Powell algorithm doesn't require gradients.  For small models it is
slow but robust.  For big models (where func and grad are simulated) with
large variance in the function estimates, this may be less robust than
the gradient-based algorithms.


## Minimizing KL divergence (cross entropy):

If you seek to maximize entropy, set `priorlogprobs` to `None`.

If you seek to minimize KL divergence between the model and a prior
density p_0, set `priorlogprobs` to an array of the log probability
densities log(p_0(x)) for each x in the sample space.  For BigModel
objects, set this to an array of the log probability densities
log(p_0(x)) for each x in the random sample from the auxiliary
distribution.

By default, use the sample matrix sampleF to estimate the entropy dual
and its gradient.  Otherwise, set self.external to the index of the
sample feature matrix in the list self.externalFs.  This applies to
'BigModel' objects only, but setting this here simplifies the code in
dual() and grad().

"""

import warnings
import re

# Make sure that DeprecationWarning within this package always gets printed
warnings.filterwarnings('always', category=DeprecationWarning,
                        module='^{0}\.'.format(re.escape(__name__)))


from .base import BaseModel
from .scipy.model import Model
from .scipy.conditionalmodel import ConditionalModel
from .scipy.bigmodel import BigModel
from .skmaxent import (FeatureTransformer,
                       MinDivergenceModel,
                       MCMinDivergenceModel)


__all__ = ['BaseModel',
           'Model',
           'ConditionalModel',
           'BigModel',
           'utils',
           'FeatureTransformer',
           'MinDivergenceModel',
           'MCMinDivergenceModel']

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = '0.4.dev0'
