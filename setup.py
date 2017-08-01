#!/usr/bin/env python

from setuptools import setup, find_packages
import versioneer

setup(
    name='maxentropy',
    version = versioneer.get_version(),
    # cmdclass=versioneer.get_cmdclass(),
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
)

