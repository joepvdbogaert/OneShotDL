#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 17:17:35 2018

@author: Joep van den Bogaert
"""
from models import OneShotTransferCNN

model = OneShotTransferCNN(log=True, verbose=0)
max_evals = 300

model.tune_with_HORD(max_evals)