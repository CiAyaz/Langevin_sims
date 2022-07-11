#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

if __name__ == "__main__":
    # cihan langevin
    setup(name='Lan_sims',
          packages=find_packages(),
          version="1.0",
          license='MIT',
          description=('A numerical solver for the Langevin equation.'),
          author="Cihan Ayaz",
          zip_safe=False,
          requires=['numpy (>=1.10.4)', 'numba (>=0.37.0)'],
          install_requires=['numpy>=1.10.4', 'numba>=0.37.0'])

