#!/usr/bin/python

import sys
import numpy as np
import numpy.linalg as la
import math
from .utils import QAM_Modulation, NLE, de2bi, _QPSK_Constellation, _16QAM_Constellation, _64QAM_Constellation

"""    Please write the code for EP algorithm with soft output detection.
"""
def EP(x,A,y,noise_var,T=10,mu=2,soft=False,pp_llr=None):  # ub as output, stable
    
    return ub, MSE
