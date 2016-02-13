#! /usr/bin/env python
# coding:utf-8


import numpy as np
from typing import Dict
from chainer import Variable

# type definition
NArray = np.ndarray
NFloatArray = np.ndarray
NIntArray = np.ndarray
State = Dict[str, Variable]
