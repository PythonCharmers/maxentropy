import numpy as np

from maxentropy.utils import dictsample


def test_dictsample():
    samplespace = np.arange(6) + 1
    samplefreq = {e: 1 / 6 for e in samplespace}
    x = dictsample(samplefreq, return_probs=None)
    assert x in samplefreq.keys()
    xs = dictsample(samplefreq, size=2, return_probs=None)
    assert xs.shape == (2,)
    for xi in xs:
        assert xi in samplefreq.keys()
