import scipy.stats
import numpy as np
import maxentropy


def test_feature_sampler():
    # Define 3 functions, vectorize them,
    # and use them with a sampler of continuous things

    def f0(x):
        return x
    
    def f1(x):
        return x**2
    
    lower, upper = (-3, 3)
    def f2(x):
        return (lower < x) & (x < upper)
    
    f = [f0, f1, f2]
    
    features = maxentropy.bigmodel.vec_feature_function(f, sparse=False)

    auxiliary = scipy.stats.norm(loc=0.0, scale=2.0)
    sampler = auxiliary_sampler_scipy(auxiliary)

    q = maxentropy.bigmodel.feature_sampler(f_vec, sampler)
    things = next(q)
    assert len(things) == 3
    assert things[0].shape == (m, n)
    assert logprob.shape == (n,)
    assert xs.ndim == 2 and xs.shape[1] == n


def test_other_stuff():
    """
    Write me!
    """
    whichplot = 1  # sub-plot in Figure 6.1 (0 , 1 , or 2)
    d = 1     # number of dimensions
    m = d*3   # number of features
    
    # Bounds
    o = np.ones(d)
    if whichplot == 0:
        lower = -2.5 * o
        upper = -lower
    elif whichplot == 1:
        lower = 0.5 * o
        upper = 2.5 * o
    elif whichplot == 2:
        lower = -0.1 * o
        upper = 0.1 * o
    x = np.linspace(-3, 3, num=10)
    
    
    # In[11]:
    
    features(x)
    
    
    # In[12]:
    
    # Target constraint values
    b = np.empty (m , float )
    if whichplot == 0:
        b [0: m :3] = 0   # expectation
        b [1: m :3] = 1   # second moment
        b [2: m :3] = 1   # truncate completely outside bounds
    elif whichplot == 1:
        b [0: m :3] = 1.0 # expectation
        b [1: m :3] = 1.2 # second moment
        b [2: m :3] = 1   # truncate completely outside bounds
    elif whichplot == 2:
        b [:] = [0. , 0.0033 , 1]
    
    
    # In[13]:
    
    b
    
    
    # Create a generator of features of random points under a Gaussian auxiliary dist $q$ with diagonal covariance matrix.
    
    # In[14]:
    
    q = maxentropy.bigmodel.feature_sampler(f_vec, sampler)
    
    
    model = maxentropy.BigModel(sampler)   # create a model
    model.verbose=True
    model.fit(f, b)                    # fit under the given constraints using SPO
    
    
    # In[ ]:
    
    get_ipython().magic('debug')
    
    
    # After running this code, the `model` has a vector
    # of parameters $\theta = (\theta_i)^{3d}_{i=1}$
    # stored as the array `model.params`. The pdf of the fitted model can then be retrieved
    # with the `model.pdf` method and plotted as follows:
    # 
    # 
    
    # In[ ]:
    
    get_ipython().magic('matplotlib inline')
    
    
    # In[ ]:
    
    # Plot the marginal pdf in dimension 0 , letting x_d =0
    # for all other dimensions d.
    xs = np.arange(lower[0], upper[0], (upper[0] - lower[0]) / 1000.)
    all_xs = np.zeros((d , len(xs)), float)
    all_xs[0, :] = xs
    pdf = model.pdf(model.features(all_xs))
    
    import matplotlib.pyplot as plt
    plt.plot(xs, pdf)
    plt.ylim(0, pdf.max()*1.1)
    
    
    # In[ ]:
    
    model.expectations()
    
    
    # In[ ]:
    
    b

