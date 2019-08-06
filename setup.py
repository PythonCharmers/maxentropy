#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='maxentropy',
    version = '0.3.0',
    packages=find_packages(),
    package_data={
        '': ['*.txt', '*.rst', '*.md'],
    },
    author='Ed Schofield',
    author_email='ed@pythoncharmers.com',
    description='Maximum entropy and minimum divergence models in Python',
    license='BSD',
    keywords='maximum-entropy minimum-divergence kullback-leibler-divergence KL-divergence bayesian-inference bayes scikit-learn sklearn prior prior-distribution',
    url='https://github.com/PythonCharmers/maxentropy.git',
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Topic :: Software Development',
                 'Topic :: Scientific/Engineering']
)

