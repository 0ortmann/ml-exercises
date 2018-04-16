# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 12:25:39 2018

@author: Felix Ortmann (0ortmann) Ina Reis (0reis)
"""
import numpy as np

a = np.arange(10)
    
b = np.full((3,3), True, dtype=bool)

c = a[a%2>0]

d = np.where(a % 2 == 0, -1, a)

e = np.zeros((3,3), int)

np.fill_diagonal(e, 1)
