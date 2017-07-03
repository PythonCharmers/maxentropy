# maxentropy: Maximum entropy and minimum divergence models in Python

## History
This package previously lived in SciPy 
(http://scipy.org) as ``scipy.maxentropy`` from versions v0.5 to v0.10. It was under-maintained and removed
from SciPy v0.11. It is now being refactored to use ``scikit-learn``'s estimator interface.

## Background

This package helps you to follow the maximum entropy principle (or the closely related principle of minimum divergence)
to construct a probability distribution (Bayesian prior) from prior information that you encode as generalized moment constraints.

The maximum entropy principle has been shown [Cox 1982, Jaynes 2003] to be the unique consistent approach to
constructing a discrete probability distribution from prior information that is available as "testable information".

If the constraints have the form of linear moment constraints:

$$
E(f_1(X)) = k_1
...
E(f_m(X)) = k_m
$$

then the principle gives rise to a unique probability distribution of **exponential form**. Most well-known probability
distributions are special cases of maximum entropy distributions. This includes uniform, geometric, exponential, Pareto,
normal, von Mises, Cauchy, and others: see https://en.wikipedia.org/wiki/Maximum_entropy_probability_distribution.

## Quickstart example: a loaded die - constructing a prior subject to known constraints

"A die is loaded so that the expectation of values on its upper face is 4.5. What is the probability distribution?"

Example use of the maximum entropy package: the unfair die example from Jaynes, *Probability Theory: The Logic of Science*, 2006.

Suppose you know that the long-run average number on the face of a 6-sided die
tossed many times is 4.5.

What probability $p(x)$ would you assign to rolling $x$ on the next roll?

<table>
<tr><th>x</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th></tr>
<tr><td>p(x)</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
</table>

Constraints:

$$
\begin{align}
E(f_1(X)) = \sum_{x=1}^6 f_1(x) p(x) &= 4.5, \text{where } f_1(x) = x \\
\text{and} \\
\sum_{x=1}^6 p(x) &= 1
\end{align}
$$

This notebook shows how to use the `maxentropy` package to find the probability distribution with maximal information entropy subject to these constraints.


```python
import numpy as np
from maxentropy.skmaxent import FeatureTransformer, MinDivergenceModel
```


```python
samplespace = np.linspace(1, 6, 6)
```


```python
def f_1(x):
    return x
```


```python
features = [f_1]
```


```python
k = np.array([4.5])
X = np.atleast_2d(k)
```


```python
X
```




    array([[ 4.5]])




```python
model = MinDivergenceModel(features, samplespace)
```


```python
len(features)
```




    1




```python
model.params
```




    array([ 0.])




```python
model.fit(X)
```




    MinDivergenceModel(algorithm='CG', features=[<function f_1 at 0x10d2a21e0>],
              format='csr_matrix', priorlogprobs=None,
              samplespace=array([ 1.,  2.,  3.,  4.,  5.,  6.]),
              vectorized=True, verbose=0)




```python
p = model.probdist()
p
```




    array([ 0.05387955,  0.07828049,  0.11373212,  0.16523906,  0.24007245,
            0.34879633])



## Minimizing KL divergence

Now we show how to construct the closest model to a given prior (in a KL divergence sense) subject to certain additional constraints.

### Example 1: constant prior
We will first give this for the example of a constant prior. (Spoiler: we will get the same model as the maximum entropy model above.)


```python
log_prior = np.log(np.ones(6)/6)
```


```python
log_prior
```




    array([-1.79175947, -1.79175947, -1.79175947, -1.79175947, -1.79175947,
           -1.79175947])




```python
np.exp(log_prior).sum()
```




    1.0000000000000002




```python
from scipy.misc import logsumexp
logsumexp(log_prior)
```




    0.0




```python
model2 = MinDivergenceModel(features, samplespace, priorlogprobs=log_prior)
```


```python
model2.fit(X)
```




    MinDivergenceModel(algorithm='CG', features=[<function f_1 at 0x10d2a21e0>],
              format='csr_matrix',
              priorlogprobs=array([-1.79176, -1.79176, -1.79176, -1.79176, -1.79176, -1.79176]),
              samplespace=array([ 1.,  2.,  3.,  4.,  5.,  6.]),
              vectorized=True, verbose=0)




```python
p2 = model2.probdist()
p2
```




    array([ 0.05387955,  0.07828049,  0.11373212,  0.16523906,  0.24007245,
            0.34879633])




```python
np.allclose(p, p2)
```




    True




```python
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
sns.barplot(np.arange(1, 7), model.probdist())
plt.title('Model 1: Probability $p(x)$ of each die face $x$')
```




    <matplotlib.text.Text at 0x11593a470>




![png](output_23_1.png)


### Example 2: Now try a different prior: $p(1) = p(2) = p(3)=p(4)=p(5)=0.1$ and $p(6) = 0.5$


```python
prior2 = np.zeros(6)
```


```python
prior2[:4] = 0.1
prior2[4] = 0.5
prior2[5] = 0.1
```


```python
prior2
```




    array([ 0.1,  0.1,  0.1,  0.1,  0.5,  0.1])




```python
prior2.sum()
```




    1.0




```python
priorlogprobs = np.log(prior2)
```


```python
priorlogprobs
```




    array([-2.30258509, -2.30258509, -2.30258509, -2.30258509, -0.69314718,
           -2.30258509])




```python
model3 = MinDivergenceModel(features, samplespace, priorlogprobs, algorithm='BFGS', verbose=False)
```

#### Before fitting the model, what do we have?


```python
model3.probdist()
```




    array([ 0.1,  0.1,  0.1,  0.1,  0.5,  0.1])




```python
np.allclose(model.probdist(), model3.probdist())
```




    False




```python
sns.barplot(np.arange(1, 7), model3.probdist())
plt.title('Model 2: Probability $p(x)$ of each die face $x$')
```




    <matplotlib.text.Text at 0x117aab208>




![png](output_35_1.png)


Are the constraints satisfied?


```python
model3.expectations()
```




    array([ 4.1])



No. (We haven't fitted the model yet.)


```python
np.allclose(model3.expectations(), k)
```




    False



#### What is the KL divergence before fitting the model ?


```python
from scipy.stats import entropy
```


```python
import numpy as np
```


```python
np.exp(model3.priorlogprobs)
```




    array([ 0.1,  0.1,  0.1,  0.1,  0.5,  0.1])




```python
model3.probdist()
```




    array([ 0.1,  0.1,  0.1,  0.1,  0.5,  0.1])




```python
model3.divergence()
```




    -3.3306690738754691e-16



Actually, this is zero, with numerical imprecision. The divergence is always >= 0 by definition.


```python
np.allclose(model3.divergence(), 0)
```




    True



#### Answer: zero


```python
# Verify with scipy.stats.entropy():
D = entropy(model3.probdist(), np.exp(model3.priorlogprobs))
np.allclose(model3.divergence(), D)
```




    True



### Now we fit the model (place constraints on it):


```python
model3.fit(X)
```




    MinDivergenceModel(algorithm='BFGS', features=[<function f_1 at 0x10d2a21e0>],
              format='csr_matrix',
              priorlogprobs=array([-2.30259, -2.30259, -2.30259, -2.30259, -0.69315, -2.30259]),
              samplespace=array([ 1.,  2.,  3.,  4.,  5.,  6.]),
              vectorized=True, verbose=False)




```python
model3.probdist()
```




    array([ 0.05125862,  0.06272495,  0.07675624,  0.09392627,  0.57468582,
            0.1406481 ])




```python
np.exp(model3.priorlogprobs)
```




    array([ 0.1,  0.1,  0.1,  0.1,  0.5,  0.1])




```python
sns.barplot(np.arange(1, 7), model3.probdist())
plt.title('Model 2: Probability $p(x)$ of each die face $x$')
```




    <matplotlib.text.Text at 0x117ce7668>




![png](output_54_1.png)
