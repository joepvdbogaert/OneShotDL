#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 17:17:35 2018

@author: Joep van den Bogaert
"""
from models import OneShotAutoencoder

model = OneShotAutoencoder(log=True, verbose=1)
max_evals = 300

model.tune_with_HORD(max_evals)