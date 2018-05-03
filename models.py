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

# use keras for the CNN
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, Reshape, UpSampling2D
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras import Model

# helper functions from OneShotDL
from helpers import load_mnist, split_and_select_random_data, reinitialize_random_weights, freeze_layers



class OneShotCNN():
    """ Class used for training a CNN for one shot image classification. 

    :param log: Boolean that indicates whether to write the results to a csv file.
    :param folds: Number of experiments to run for each combination of hyperparameter values.
    :param verbose: Whether to print detailed information to the console.
    """

    name = "OneShotCNN"

    def __init__(self, log=False, folds=10, batchsize=128, verbose=1):

        # specify the parameter ranges as [min, max].
        # first continuous, then integer params.
        self.rgs = {'learning_rate': [0.0001, 0.005],
                    'dropout_rate1': [0.0, 0.6],
                    'dropout_rate2': [0.0, 0.7],
                    'width_shift': [0.0, 0.6],
                    'height_shift': [0.0, 0.6],
                    'shear': [0.0, 0.7],
                    'zoom': [0.0, 0.6],
                    'num_conv_layers': [2, 5],
                    'num_dense_layers': [0, 3],
                    'num_maxpools': [0, 2],
                    'neurons_first_conv': [3, 6],
                    'neurons_remaining_conv': [4, 7],
                    'neurons_dense': [5, 10],
                    'rotation': [0, 360],
                    'horizontal_flip': [0, 1],
                    'epochs': [50, 3000]}

        self.hyperparams = list(self.rgs.keys())
        self.dim = len(self.hyperparams)
        self.hyper_map = {self.hyperparams[i]:i for i in range(len(self.rgs.keys()))}
        self.xlow = np.array([self.rgs[key][0] for key in self.hyperparams])
        self.xup = np.array([self.rgs[key][1] for key in self.hyperparams])
        self.continuous = np.arange(0, 7)
        self.integer = np.arange(7, self.dim)

        # fixed parameters
        self.batchsize = batchsize
        self.log = log
        self.nfolds = folds # for cross validation

        # data
        self.x_train, self.y_train = load_mnist("./Data/", kind='train')
        self.x_test, self.y_test = load_mnist("./Data/", kind='test')
        self.num_classes = self.y_test.shape[1]

        # logging results
        self.param_log = pd.DataFrame(columns=self.hyperparams)
        self.scores_log = pd.DataFrame(columns=np.arange(1,self.nfolds+1))

        # printing
        self.verbose = verbose

        # counter
        self.exp_number = 0


    def objfunction(self, params):
        """ The overall objective function to provide to pySOT's black box optimization. 

        :param params: The parameters to use for the function evaluation (array like).
        :returns: Negative mean accuracy on the test set (negative for minimization).
        """

        self.exp_number += 1
        print("-------------\nExperiment {}.\n-------------".format(self.exp_number))
        if self.verbose:
            for p in self.hyperparams:
                print(p+": "+str(params[self.hyper_map[p]]))
            print("-------------")


        def define_model(params):
            """ Creates the Keras model based on given parameters. 

            :param params: The parameters of the object indicating the architecture of the NN.
            :returns: A tuple of the model and datagenerator. 
            """

            # determine where to put max pools:
            convs = int(params[self.hyper_map['num_conv_layers']])
            pools = int(params[self.hyper_map['num_maxpools']])
            add_max_pool_after_layer = np.arange(convs)[::-1][0:pools]

            # initialize model
            model = Sequential()

            # add first convolutional layer and specify input shape
            model.add(Conv2D(2**int(params[self.hyper_map['neurons_first_conv']]), 
                             kernel_size=(3,3), activation='relu',
                             input_shape=(28,28,1), data_format="channels_last"))

            # possibly add max pool
            if 0 in add_max_pool_after_layer:
                model.add(MaxPooling2D(pool_size=(2, 2)))

            # possibly add more conv layers
            if convs > 1:

                for l in range(1, convs):

                    model.add(Conv2D(2**int(params[self.hyper_map['neurons_remaining_conv']]), (3, 3), activation='relu'))

                    if l in add_max_pool_after_layer:
                        model.add(MaxPooling2D(pool_size=(2, 2)))

            # dropout and flatten before the dense layers
            model.add(Dropout(params[self.hyper_map['dropout_rate1']]))
            model.add(Flatten())

            # add dense layers before the classification layer
            for l in range(int(params[self.hyper_map['num_dense_layers']])):
                model.add(Dense(2**int(params[self.hyper_map['neurons_dense']]), activation='relu'))

            # classification layer
            model.add(Dropout(params[self.hyper_map['dropout_rate2']]))
            model.add(Dense(self.num_classes, activation='softmax'))

            # compile and return
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.RMSprop(lr=params[self.hyper_map['learning_rate']]),
                          metrics=['accuracy'])

            # create data generator with augmentations
            datagen = ImageDataGenerator(width_shift_range=params[self.hyper_map['width_shift']],
                                         height_shift_range=params[self.hyper_map['height_shift']],
                                         shear_range=params[self.hyper_map['shear']],
                                         zoom_range=params[self.hyper_map['zoom']],
                                         horizontal_flip=params[self.hyper_map['horizontal_flip']],
                                         rotation_range=params[self.hyper_map['rotation']])

            return model, datagen


        def cross_validate(x, y, xtest, ytest, params, n):
            """ Cross validate with random sampling. 

            :param x: the training data (a 4D ndarray).
            :param y: the training labels (one hot encoded).
            :param xtest: the test data (a 4D ndarray).
            :param ytest: the test labels (one hot encoded).
            :param params: the parameters that define the model (array like).
            :param n: the number of experiments to run for the given parameters.
            :returns: a list of n test accuracies.
            """

            print("Cross validating in {} folds..".format(self.nfolds))
            scores = []

            for i in range(n):

                # early elimination of candidate solution
                if i > 1 and np.mean(scores) < 0.21:
                    break

                # get data
                x_target_labeled, y_target, x_test, y_test, _, _, _ = \
                    split_and_select_random_data(x, y, xtest, ytest, num_target_classes=5, num_examples_per_class=1)

                # define model according to parameters
                model, datagen = define_model(params)

                # fits the model on batches with real-time data augmentation:
                print("fit {}:".format(i+1))
                model.fit_generator(datagen.flow(x_target_labeled, y_target, batch_size=x_target_labeled.shape[0]),
                                    steps_per_epoch=1, epochs=params[self.hyper_map['epochs']], verbose=0)

                # evaluate on test set
                loss, accuracy = model.evaluate(x_test, y_test, verbose=0, batch_size=y.shape[0])

                # report score
                print("test accuracy: {}%.".format(round(accuracy*100, 2)))
                scores.append(accuracy)

                # prevent memory buildup
                K.clear_session()

            return scores


        # run the experiment nfolds times and print mean and std.
        scores = cross_validate(self.x_train, self.y_train, self.x_test, self.y_test, params, self.nfolds)
        print("Scores: {}.\nMean: {}%. Standard deviation: {}%".format(scores, round(np.mean(scores)*100, 2), round(np.std(scores)*100, 2)))

        # log after every function call / for every set of parameters
        if self.log:
            # add zeros to scores if candidate was eliminated prematurely to ensure succesful logging
            for i in np.arange(self.nfolds - len(scores)):
                scores.append( np.float(0.0) )
            # log to object and to files
            self.param_log = pd.concat([self.param_log, pd.DataFrame(np.reshape(params, (1,self.dim)), columns=self.hyperparams)])
            self.scores_log = pd.concat([self.scores_log, pd.DataFrame(np.reshape(scores, (1,self.nfolds)), columns=np.arange(1,self.nfolds+1))])
            self.param_log.to_csv("./Results/"+self.name+"_params_log.csv", index=False)
            self.scores_log.to_csv("./Results/"+self.name+"_scores_log.csv", index=False)

        # prevent memory buildup
        K.clear_session()

        # return negative mean value for pySOT to minimize
        return -np.mean(scores)


    def tune_with_HORD(self, max_evaluations):
        """ Automatically tune hyperparameters using HORD (Ilievski et al., 2017). 

        :param max_evaluations: maximum function evaluations (so maximum number of parameter settings to try).
        """
        
        # create controller
        controller = SerialController(self.objfunction)
        # experiment design
        exp_des = LatinHypercube(dim=self.dim, npts=2*self.dim+1)
        # use a cubic RBF interpolant with a linear tail
        surrogate = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=max_evaluations)
        # use DYCORS with 100d candidate points
        adapt_samp = CandidateDYCORS(data=self, numcand=100*self.dim)
        # strategy
        strategy = SyncStrategyNoConstraints(worker_id=0, data=self, maxeval=max_evaluations, nsamples=1,
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



class OneShotTransferCNN():
    """ Class used for one shot image classification with Transfer Learning. 

    :param log: Boolean that indicates whether to write the results to a csv file.
    :param folds: Number of experiments to run for each combination of hyperparameter values.
    :param verbose: Whether to print detailed information to the console.
    """
    
    name = "OneShotTransferCNN"

    def __init__(self, log=False, folds=10, batchsize=128, verbose=1):
        
        # specify the parameter ranges as [min, max].
        # first continuous, then integer params.
        self.rgs = {'learning_rate': [0.0001, 0.005],
                    'finetune_learning_rate': [0.0001, 0.01],
                    'dropout_rate1': [0.0, 0.6],
                    'dropout_rate2': [0.0, 0.7],
                    'width_shift': [0.0, 0.6],
                    'height_shift': [0.0, 0.6],
                    'shear': [0.0, 0.7],
                    'zoom': [0.0, 0.6],
                    'finetune_width_shift': [0.0, 0.4],
                    'finetune_height_shift': [0.0, 0.4],
                    'finetune_shear': [0.0, 0.7],
                    'finetune_zoom': [0.0, 0.4],
                    'num_conv_layers': [2, 5],
                    'num_dense_layers': [0, 3],
                    'num_maxpools': [0, 2],
                    'neurons_first_conv': [3, 6],
                    'neurons_remaining_conv': [4, 7],
                    'neurons_dense': [5, 10],
                    'rotation': [0, 360],
                    'finetune_rotation': [0, 360],
                    'horizontal_flip': [0, 1],
                    'finetune_horizontal_flip': [0, 1],
                    'epochs': [3, 10],
                    'finetune_epochs': [100, 2000],
                    'num_fixed_layers': [1, 13],
                    'reinitialize_weights': [0, 1]}

        self.hyperparams = list(self.rgs.keys())
        self.dim = len(self.hyperparams)
        self.hyper_map = {self.hyperparams[i]:i for i in range(len(self.rgs.keys()))}
        self.xlow = np.array([self.rgs[key][0] for key in self.hyperparams])
        self.xup = np.array([self.rgs[key][1] for key in self.hyperparams])
        self.continuous = np.arange(0, 12)
        self.integer = np.arange(12, self.dim)

        # fixed parameters
        self.batchsize = batchsize
        self.log = log
        self.nfolds = folds # for cross validation

        # data
        self.x_train, self.y_train = load_mnist("./Data/", kind='train')
        self.x_test, self.y_test = load_mnist("./Data/", kind='test')
        self.num_classes = self.y_test.shape[1]

        # logging results
        self.param_log = pd.DataFrame(columns=self.hyperparams)
        self.scores_log = pd.DataFrame(columns=np.arange(1,self.nfolds+1))

        # printing
        self.verbose = verbose

        # counter
        self.exp_number = 0


    def objfunction(self, params):
        """ The overall objective function to provide to pySOT's black box optimization. 

        :param params: The parameters to use for the function evaluation (array like).
        :returns: Negative mean accuracy on the test set (negative for minimization).
        """

        self.exp_number += 1
        print("-------------\nExperiment {}.\n-------------".format(self.exp_number))
        if self.verbose:
            for p in self.hyperparams:
                print(p+": "+str(params[self.hyper_map[p]]))
            print("-------------")


        def define_model(params):
            """ Creates the Keras model based on given parameters. 

            :param params: The parameters of the object indicating the architecture of the NN.
            :returns: A tuple of the model and datagenerator. 
            """

            # determine where to put max pools:
            convs = int(params[self.hyper_map['num_conv_layers']])
            pools = int(params[self.hyper_map['num_maxpools']])
            add_max_pool_after_layer = np.arange(convs)[::-1][0:pools]

            print("Conv layers: {}. Max pools: {}. Positioned after layers: {}.".format(
                  convs, pools, add_max_pool_after_layer))

            # initialize model
            model = Sequential()

            # add first convolutional layer and specify input shape
            model.add(Conv2D(2**int(params[self.hyper_map['neurons_first_conv']]), 
                             kernel_size=(3,3), activation='relu', 
                             input_shape=(28,28,1), data_format="channels_last"))

            # possibly add max pool
            if 0 in add_max_pool_after_layer:
                model.add(MaxPooling2D(pool_size=(2, 2)))

            # possibly add more conv layers
            if convs > 1:

                for l in range(1, convs):

                    model.add(Conv2D(2**int(params[self.hyper_map['neurons_remaining_conv']]), (3, 3), activation='relu'))

                    if l in add_max_pool_after_layer:
                        model.add(MaxPooling2D(pool_size=(2, 2)))

            # dropout and flatten before the dense layers
            model.add(Dropout(params[self.hyper_map['dropout_rate1']]))
            model.add(Flatten())

            # add dense layers before the classification layer
            for l in range(int(params[self.hyper_map['num_dense_layers']])):
                model.add(Dense(2**int(params[self.hyper_map['neurons_dense']]), activation='relu'))

            # classification layer
            model.add(Dropout(params[self.hyper_map['dropout_rate2']]))
            model.add(Dense(self.num_classes, activation='softmax'))

            # compile and return
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.RMSprop(lr=params[self.hyper_map['learning_rate']]),
                          metrics=['accuracy'])

            # create data generator with augmentations
            datagen = ImageDataGenerator(width_shift_range=params[self.hyper_map['width_shift']],
                                         height_shift_range=params[self.hyper_map['height_shift']],
                                         shear_range=params[self.hyper_map['shear']],
                                         zoom_range=params[self.hyper_map['zoom']],
                                         horizontal_flip=params[self.hyper_map['horizontal_flip']],
                                         rotation_range=params[self.hyper_map['rotation']])

            return model, datagen


        def cross_validate(x, y, xtest, ytest, params, n):
            """ Cross validate with random sampling. 

            :param x: the training data (a 4D ndarray).
            :param y: the training labels (one hot encoded).
            :param xtest: the test data (a 4D ndarray).
            :param ytest: the test labels (one hot encoded).
            :param params: the parameters that define the model (array like).
            :param n: the number of experiments to run for the given parameters.
            :returns: a list of n test accuracies.
            """

            print("Cross validating in {} folds..".format(self.nfolds))
            scores = []

            for i in range(n):

                # early elimination of candidate solution
                if i > 1 and np.mean(scores) < 0.21:
                    break

                # get data
                x_target_labeled, y_target, x_test, y_test, _, x_auxiliary, y_auxiliary = \
                    split_and_select_random_data(x, y, xtest, ytest, num_target_classes=5, num_examples_per_class=1)

                # define model according to parameters
                model, datagen = define_model(params)

                # fit the model on batches with real-time data augmentation:
                print("fit {}:".format(i+1))
                model.fit_generator(datagen.flow(x_auxiliary, y_auxiliary, batch_size=self.batchsize),
                                                 steps_per_epoch=int(round(x_auxiliary.shape[0]/self.batchsize, 0)),
                                                 epochs=params[self.hyper_map['epochs']], verbose=self.verbose)

                # freeze layers reinitialize the last layer and possibly reinitialize the rest
                num_fixed = int(min(params[self.hyper_map['num_fixed_layers']], len(model.layers)))
                model = freeze_layers(model, 
                                      num_fixed,
                                      count_only_trainable_layers=True,
                                      reinitialize_remaining=params[self.hyper_map['reinitialize_weights']],
                                      reinitialize_last=True)

                # recompile with the frozen layers
                model.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer=keras.optimizers.RMSprop(lr=params[self.hyper_map['finetune_learning_rate']]),
                              metrics=['accuracy'])

                # check if it does what you think
                if self.verbose:
                    print(model.summary())
                    print("Number of fixed layers: {}".format(params[self.hyper_map['num_fixed_layers']]))

                # fine tune the model
                finetune_datagen = ImageDataGenerator(width_shift_range=params[self.hyper_map['finetune_width_shift']],
                                                      height_shift_range=params[self.hyper_map['finetune_height_shift']],
                                                      shear_range=params[self.hyper_map['finetune_shear']],
                                                      zoom_range=params[self.hyper_map['finetune_zoom']],
                                                      horizontal_flip=params[self.hyper_map['finetune_horizontal_flip']],
                                                      rotation_range=params[self.hyper_map['finetune_rotation']])

                model.fit_generator(finetune_datagen.flow(x_target_labeled, y_target, batch_size=x_target_labeled.shape[0]),
                                                          steps_per_epoch=1, epochs=params[self.hyper_map['finetune_epochs']],
                                                          verbose=self.verbose)

                # evaluate on test set
                loss, accuracy = model.evaluate(x_test, y_test, verbose=self.verbose, batch_size=y.shape[0])

                # report score
                print("test accuracy: {}%.".format(round(accuracy*100, 2)))
                scores.append(accuracy)

                # prevent memory buildup
                K.clear_session()

            return scores


        # run the experiment nfolds times and print mean and std.
        scores = cross_validate(self.x_train, self.y_train, self.x_test, self.y_test, params, self.nfolds)

        print("Scores: {}.\nMean: {}%. Standard deviation: {}%".format(scores, round(np.mean(scores)*100, 2), round(np.std(scores)*100, 2)))

        # log after every function call / for every set of parameters
        if self.log:
            # add zeros to scores if candidate was eliminated prematurely to ensure succesful logging
            for i in np.arange(self.nfolds - len(scores)):
                scores.append( np.float(0.0) )
            # log to object and to files
            self.param_log = pd.concat([self.param_log, pd.DataFrame(np.reshape(params, (1,self.dim)), columns=self.hyperparams)])
            self.scores_log = pd.concat([self.scores_log, pd.DataFrame(np.reshape(scores, (1,self.nfolds)), columns=np.arange(1,self.nfolds+1))])
            self.param_log.to_csv("./Results/"+self.name+"_params_log.csv", index=False)
            self.scores_log.to_csv("./Results/"+self.name+"_scores_log.csv", index=False)

        # prevent memory buildup
        K.clear_session()

        # Return negative mean value for pySOT to minimize
        return -np.mean(scores)


    def tune_with_HORD(self, max_evaluations):
        """ Automatically tune hyperparameters using HORD (Ilievski et al., 2017). 

        :param max_evaluations: maximum function evaluations (so maximum number of parameter settings to try).
        """

        # create controller
        controller = SerialController(self.objfunction)
        # experiment design
        exp_des = LatinHypercube(dim=self.dim, npts=2*self.dim+1)
        # use a cubic RBF interpolant with a linear tail
        surrogate = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=max_evaluations)
        # use DYCORS with 100d candidate points
        adapt_samp = CandidateDYCORS(data=self, numcand=100*self.dim)
        # strategy
        strategy = SyncStrategyNoConstraints(worker_id=0, data=self, maxeval=max_evaluations, nsamples=1,
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



class OneShotAutoencoder():
    """ Class used for one shot image classification using an Autoencoder. 

    :param log: Boolean that indicates whether to write the results to a csv file.
    :param folds: Number of experiments to run for each combination of hyperparameter values.
    :param verbose: Whether to print detailed information to the console.
    """

    name = "OneShotAutoencoder"

    def __init__(self, log=False, folds=10, batchsize=128, verbose=1):

        # specify the parameter ranges as [min, max].
        # first continuous, then integer params.
        self.rgs = {'learning_rate': [0.0001, 0.05],
                    'finetune_learning_rate': [0.0001, 0.01],
                    'dropout_rate1': [0.0, 0.6],
                    'dropout_rate2': [0.0, 0.7],
                    'width_shift': [0.0, 0.6],
                    'height_shift': [0.0, 0.6],
                    'shear': [0.0, 0.7],
                    'zoom': [0.0, 0.6],
                    'finetune_width_shift': [0.0, 0.4],
                    'finetune_height_shift': [0.0, 0.4],
                    'finetune_shear': [0.0, 0.7],
                    'finetune_zoom': [0.0, 0.4],
                    'num_conv_layers': [2, 5],
                    'num_dense_layers': [1, 3],
                    'num_maxpools': [0, 2],
                    'neurons_first_conv': [3, 6],
                    'neurons_remaining_conv': [4, 7],
                    'neurons_dense': [5, 10],
                    'rotation': [0, 360],
                    'finetune_rotation': [0, 360],
                    'horizontal_flip': [0, 1],
                    'finetune_horizontal_flip': [0, 1],
                    'epochs': [3, 10],
                    'finetune_epochs': [100, 1000],
                    'num_fixed_layers': [0, 8],
                    'reinitialize_weights': [0, 1],
                    'neurons_bottleneck': [8, 128]}

        self.hyperparams = list(self.rgs.keys())
        self.dim = len(self.hyperparams)
        self.hyper_map = {self.hyperparams[i]:i for i in range(len(self.rgs.keys()))}
        self.xlow = np.array([self.rgs[key][0] for key in self.hyperparams])
        self.xup = np.array([self.rgs[key][1] for key in self.hyperparams])
        self.continuous = np.arange(0, 12)
        self.integer = np.arange(12, self.dim)

        # fixed parameters
        self.batchsize = batchsize
        self.log = log
        self.nfolds = folds # for cross validation

        # data
        self.x_train, self.y_train = load_mnist("./Data/", kind='train')
        self.x_test, self.y_test = load_mnist("./Data/", kind='test')
        self.num_classes = self.y_test.shape[1]

        # logging results
        self.param_log = pd.DataFrame(columns=self.hyperparams)
        self.scores_log = pd.DataFrame(columns=np.arange(1,self.nfolds+1))

        # printing
        self.verbose = verbose

        # counter
        self.exp_number = 0


    def objfunction(self, params):
        """ The overall objective function to provide to pySOT's black box optimization. 

        :param params: The parameters to use for the function evaluation (array like).
        :returns: Negative mean accuracy on the test set (negative for minimization).
        """

        self.exp_number += 1
        print("-------------\nExperiment {}.\n-------------".format(self.exp_number))
        if self.verbose:
            for p in self.hyperparams:
                print(p+": "+str(params[self.hyper_map[p]]))
            print("-------------")


        def define_model(params):
            """ Creates the Keras model based on given parameters. 

            :param params: The parameters of the object indicating the architecture of the NN.
            :returns: A tuple of the model and datagenerator. 
            """

            # determine where to put max pools:
            convs = int(params[self.hyper_map['num_conv_layers']])
            pools = int(params[self.hyper_map['num_maxpools']])
            add_max_pool_after_layer = np.arange(convs)[::-1][0:pools]
            add_upsampling_before_layer = np.arange(convs)[0:pools]
            print("convs: {}. pools: {}. max pools after layers {}. upsampling after: {} (zero-based indexing).".format(
                  convs, pools, add_max_pool_after_layer, add_upsampling_before_layer))

            # initialize model
            input_img = Input(shape=(28, 28, 1))

            # add first convolutional layer and specify input shape
            e = Conv2D(2**int(params[self.hyper_map['neurons_first_conv']]), 
                       kernel_size=(3,3), activation='relu', padding='same')(input_img)

            # possibly add max pool
            if 0 in add_max_pool_after_layer:
                e = MaxPooling2D(pool_size=(2, 2))(e)

            # possibly add more conv layers
            if convs > 1:

                for l in range(1, convs):
                    e = Conv2D(2**int(params[self.hyper_map['neurons_remaining_conv']]), (3, 3),
                               activation='relu', padding='same')(e)

                    if l in add_max_pool_after_layer:
                        e = MaxPooling2D(pool_size=(2, 2))(e)

            # remember the shape before flattening
            reshape_shape = K.int_shape(e)[1:]

            # flatten before the dense layers
            e = Flatten()(e)

            # add dense layers before the classification layer
            for l in range(int(params[self.hyper_map['num_dense_layers']])):
                e = Dense(2**int(params[self.hyper_map['neurons_dense']]), activation='relu')(e)

            # add bottleneck layer
            encoder = Dense(int(params[self.hyper_map['neurons_bottleneck']]), activation='relu')(e)

            # add dense layers of decoding part
            for l in range(int(params[self.hyper_map['num_dense_layers']])):
                if l == 0:
                    e = Dense(2**int(params[self.hyper_map['neurons_dense']]), activation='relu')(encoder)
                else:
                    e = Dense(2**int(params[self.hyper_map['neurons_dense']]), activation='relu')(e)


            e = Dense(reshape_shape[0]*reshape_shape[1]*reshape_shape[2], activation='relu')(e)

            # reshape to image format
            e = Reshape(reshape_shape)(e)

            # add conv layers and upsampling mirroring the encoder
            for l in range(int(params[self.hyper_map['num_conv_layers']])):

                if l in add_upsampling_before_layer:
                    e = UpSampling2D(size=(2, 2))(e)

                if l == int(params[self.hyper_map['num_conv_layers']]-1):
                    neurons = 2**int(params[self.hyper_map['neurons_first_conv']])
                else:
                    neurons = 2**int(params[self.hyper_map['neurons_remaining_conv']])

                e = Conv2D(neurons, (3, 3), activation='relu', padding='same')(e)

            # add classification layer
            decoder = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(e)

            # initialize model
            autoencoder = Model(input_img, decoder)

            autoencoder.compile(optimizer=keras.optimizers.RMSprop(lr=params[self.hyper_map['learning_rate']]),
                                loss='binary_crossentropy', metrics=['accuracy'])

            # create data generator with augmentations
            datagen = ImageDataGenerator(width_shift_range=params[self.hyper_map['width_shift']],
                                         height_shift_range=params[self.hyper_map['height_shift']],
                                         shear_range=params[self.hyper_map['shear']],
                                         zoom_range=params[self.hyper_map['zoom']],
                                         horizontal_flip=params[self.hyper_map['horizontal_flip']],
                                         rotation_range=params[self.hyper_map['rotation']])

            # check model to see if makes sense
            if self.verbose:
                print(autoencoder.summary())

            return autoencoder, encoder, datagen, input_img


        def cross_validate(x, y, xtest, ytest, params, n):
            """ Cross validate with random sampling. 

            :param x: the training data (a 4D ndarray).
            :param y: the training labels (one hot encoded).
            :param xtest: the test data (a 4D ndarray).
            :param ytest: the test labels (one hot encoded).
            :param params: the parameters that define the model (array like).
            :param n: the number of experiments to run for the given parameters.
            :returns: a list of n test accuracies.
            """

            print("Cross validating in {} folds..".format(self.nfolds))
            scores = []

            for i in range(n):

                # early elimination of candidate solution
                if i > 1 and np.mean(scores) < 0.21:
                    break

                # get data
                x_target_labeled, y_target, x_test, y_test, x_target_unlabeled, _, _ = \
                    split_and_select_random_data(x, y, xtest, ytest, num_target_classes=5, num_examples_per_class=1)

                # define model according to parameters
                autoencoder, encoder, datagen, input_img = define_model(params)
                
                # fit the model on batches with real-time data augmentation:
                print("fit {}:".format(i+1))
                autoencoder.fit_generator(datagen.flow(x_target_unlabeled, x_target_unlabeled, batch_size=self.batchsize),
                                          steps_per_epoch=int(round(x_target_unlabeled.shape[0]/self.batchsize, 0)),
                                          epochs=params[self.hyper_map['epochs']], verbose=self.verbose)

                # initialize classification model
                classifier = Dense(y.shape[1], activation='softmax')(encoder)
                model = Model(input_img, classifier)

                # freeze layers, possibly reinitialize other layers 
                # (except the last one because it is just created)
                num_fixed = int(min(params[self.hyper_map['num_fixed_layers']], len(model.layers)))
                model = freeze_layers(model,
                                      num_fixed,
                                      count_only_trainable_layers=True,
                                      reinitialize_remaining=params[self.hyper_map['reinitialize_weights']],
                                      reinitialize_last=False)

                # compile
                model.compile(optimizer=keras.optimizers.RMSprop(lr=params[self.hyper_map['finetune_learning_rate']]),
                              loss=keras.losses.categorical_crossentropy,
                              metrics=['accuracy'])

                # check if it does what you think
                if self.verbose:
                    print(model.summary())
                    print("Number of fixed layers: {}".format(params[self.hyper_map['num_fixed_layers']]))

                # fine tune the model
                finetune_datagen = ImageDataGenerator(width_shift_range=params[self.hyper_map['finetune_width_shift']],
                                                      height_shift_range=params[self.hyper_map['finetune_height_shift']],
                                                      shear_range=params[self.hyper_map['finetune_shear']],
                                                      zoom_range=params[self.hyper_map['finetune_zoom']],
                                                      horizontal_flip=params[self.hyper_map['finetune_horizontal_flip']],
                                                      rotation_range=params[self.hyper_map['finetune_rotation']])

                model.fit_generator(finetune_datagen.flow(x_target_labeled,
                                                          y_target,
                                                          batch_size=x_target_labeled.shape[0]),
                                    steps_per_epoch=1,
                                    epochs=params[self.hyper_map['finetune_epochs']],
                                    verbose=self.verbose)

                # evaluate on test set
                loss, accuracy = model.evaluate(x_test, y_test, verbose=self.verbose, batch_size=y.shape[0])

                # report score
                print("test accuracy: {}%.".format(round(accuracy*100, 2)))
                scores.append(accuracy)

                # prevent memory buildup
                K.clear_session()

            return scores


        # run the experiment nfolds times and print mean and std.
        scores = cross_validate(self.x_train, self.y_train, self.x_test, self.y_test, params, self.nfolds)

        print("Scores: {}.\nMean: {}%. Standard deviation: {}%".format(
              scores, round(np.mean(scores)*100, 2), round(np.std(scores)*100, 2)))

        # log after every function call / for every set of parameters
        if self.log:
            # add zeros to scores if candidate was eliminated prematurely to ensure succesful logging
            for i in np.arange(self.nfolds - len(scores)):
                scores.append( np.float(0.0) )
            # log to object and to files
            self.param_log = pd.concat([self.param_log, pd.DataFrame(np.reshape(params, (1,self.dim)), columns=self.hyperparams)])
            self.scores_log = pd.concat([self.scores_log, pd.DataFrame(np.reshape(scores, (1,self.nfolds)), columns=np.arange(1,self.nfolds+1))])
            self.param_log.to_csv("./Results/"+self.name+"_params_log.csv", index=False)
            self.scores_log.to_csv("./Results/"+self.name+"_scores_log.csv", index=False)

        # prevent memory buildup
        K.clear_session()

        # Return negative mean value for pySOT to minimize
        return -np.mean(scores)


    def tune_with_HORD(self, max_evaluations):
        """ Automatically tune hyperparameters using HORD (Ilievski et al., 2017). 

        :param max_evaluations: maximum function evaluations (so maximum number of parameter settings to try).
        """

        # create controller
        controller = SerialController(self.objfunction)
        # experiment design
        exp_des = LatinHypercube(dim=self.dim, npts=2*self.dim+1)
        # use a cubic RBF interpolant with a linear tail
        surrogate = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=max_evaluations)
        # use DYCORS with 100d candidate points
        adapt_samp = CandidateDYCORS(data=self, numcand=100*self.dim)
        # strategy
        strategy = SyncStrategyNoConstraints(worker_id=0, data=self, maxeval=max_evaluations, nsamples=1,
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