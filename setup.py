#!/usr/bin/env python

import io
import re

from setuptools import setup

with io.open("maxentropy/__init__.py", "rt", encoding="utf8") as f:
    version = re.search(r"__version__ = '(.*?)'", f.read()).group(1)

from os import path

this_directory = path.abspath(path.dirname(__file__))

with io.open(path.join(this_directory, 'README.md'), encoding='utf-8') as file:
    long_description = file.read()


setup(
    name='maxentropy',
    version = '0.3.0',
    packages=['maxentropy', 'maxentropy.scipy'],
    package_data={
        '': ['*.txt', '*.rst', '*.md'],
        'examples': ['*.py'],
        'notebooks': ['*.ipynb']
    },
    install_requires=[
        'numpy',
        'scipy',
        'six',
        'sklearn',
    ],
    author='Ed Schofield',
    author_email='ed@pythoncharmers.com',
    description='Maximum entropy and minimum divergence models in Python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='BSD',
    keywords='maximum-entropy minimum-divergence kullback-leibler-divergence KL-divergence bayesian-inference bayes scikit-learn sklearn prior prior-distribution',
    url='https://github.com/PythonCharmers/maxentropy.git',
    python_requires=">=3.3",
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

