#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 17:17:35 2018

@author: Joep van den Bogaert
"""
from models import OneShotCNN
model = OneShotCNN(log=True, verbose=0)
model.tune_with_HORD(200)