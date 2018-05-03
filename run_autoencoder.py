#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 17:17:35 2018

@author: Joep van den Bogaert
"""
import resource
resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

from models import OneShotAutoencoder

model = OneShotAutoencoder(log=True, verbose=1)
max_evals = 200

model.tune_with_HORD(max_evals)