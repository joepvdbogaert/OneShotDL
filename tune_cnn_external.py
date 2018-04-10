# -*- coding: utf-8 -*-
"""
Developed in Feb and Mar, 2018

One Shot Image Classification on Zalando MNIST

@author: Joep van den Bogaert
"""

# imports
import numpy as np
import pandas as pd
from pySOT import SyncStrategyNoConstraints, LatinHypercube, RBFInterpolant, CandidateDYCORS, CubicKernel, LinearTail
from threading import Thread, current_thread
from datetime import datetime
from poap.controller import ThreadController, BasicWorkerThread, SerialController

from models import OneShotCNN

max_evaluations = 200
model = OneShotCNN(log=True, verbose=1)

# create controller
controller = SerialController(model.objfunction)
# experiment design
exp_des = LatinHypercube(dim=model.dim, npts=2*model.dim+1)
# use a cubic RBF interpolant with a linear tail
surrogate = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=max_evaluations)
# use DYCORS with 100d candidate points
adapt_samp = CandidateDYCORS(data=model, numcand=100*model.dim)
# strategy
strategy = SyncStrategyNoConstraints(worker_id=0, data=model, maxeval=max_evaluations, nsamples=1,
                                     exp_design=exp_des, response_surface=surrogate,
                                     sampling_method=adapt_samp)
controller.strategy = strategy

# Run the optimization strategy
start_time = datetime.now()
result = controller.run()

print('Best value found: {0}'.format(result.value))
print('Best solution found: {0}\n'.format(
    np.array_str(result.params[0], max_line_width=np.inf,
                 precision=5, suppress_small=True)))

print('Started: '+str(start_time)+'. Ended: ' + str(datetime.now()))