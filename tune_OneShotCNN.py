# imports
import numpy as np
from pySOT import SyncStrategyNoConstraints, LatinHypercube, RBFInterpolant, CandidateDYCORS, CubicKernel, LinearTail
from threading import Thread, current_thread
from datetime import datetime
from poap.controller import ThreadController, BasicWorkerThread, SerialController

# use keras for the cnn tuning example
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

# OneShotDL
from helpers import load_mnist, split_and_select_random_data
from models import OneShotCNN

data = OneShotCNN(log=True)

maxeval = 200

# create the controller
controller = SerialController(data.objfunction)
# experiment design
exp_des = LatinHypercube(dim=data.dim, npts=2*data.dim+1)
# Use a cubic RBF interpolant with a linear tail
surrogate = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=maxeval)
# Use DYCORS with 100d candidate points
adapt_samp = CandidateDYCORS(data=data, numcand=100*data.dim)

strategy = SyncStrategyNoConstraints(worker_id=0, data=data, maxeval=maxeval, nsamples=1,
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

millis = int(round(time.time() * 1000))
print('Started: '+str(start_time)+'. Ended: ' + str(datetime.now()) + ' (' + str(millis) + ')')