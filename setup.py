# This file is part of the Extra-P Adaptive Modeler software (https://github.com/extra-p/extrap-adaptive-modeler)
#
# Copyright (c) 2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

info = {}
with open("extrap_adaptive_modeler/__init__.py") as fp:
    exec(fp.read(), info)

setup(
    packages=find_packages(include=('extrap_adaptive_modeler', 'extrap_adaptive_modeler.*')),
    name="extrap-adaptive-modeler",
    version=info['__version__'],
    author="Extra-P project",
    author_email="extra-p@lists.parallel.informatik.tu-darmstadt.de",
    description=info['__description__'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/extra-p/extrap-adaptive-modeler",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.8',
    include_package_data=True,
    install_requires=[
        "extrap~=4.2",
        "tensorflow~=2.9",
        "importlib-resources>=6.1.1"
    ],
)
