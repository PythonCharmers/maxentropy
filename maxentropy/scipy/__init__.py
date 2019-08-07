"""
Old interface from scipy.maxentropy.

This is deprecated in favour of a new scikit-learn compatible API.
"""


from .basemodel import BaseModel
from .model import Model
from .conditionalmodel import ConditionalModel
from .bigmodel import BigModel


__all__ = ['BaseModel',
           'Model',
           'ConditionalModel',
           'BigModel',
           'FeatureTransformer',
           'MinDivergenceModel',
           'MCMinDivergenceModel']


