# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26, 2017

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
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

# helper functions from OneShotDL
from helpers import load_mnist, split_and_select_random_data


class OneShotCNN():
    """ Class used for training a CNN for one shot image classification. 

    :param log: Boolean that indicates whether to write the results to a csv file.
    :param folds: Number of experiments to run for each combination of hyperparameter values.
    :param verbose: Whether to print detailed information to the console.
    """

    def __init__(self, log=False, folds=10, batchsize=128, verbose=True):
        
        # specify the parameter ranges as [min, max].
        # first continuous, then integer params.
        self.rgs = {'learning_rate': [0.0001, 0.01],
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
                    'epochs': [50, 1000]}

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


        def define_model(params):
            """ Creates the Keras model based on given parameters. 

            :param params: The parameters of the object indicating the architecture of the NN.
            :returns: A tuple of the model and datagenerator. 
            """

            # determine where to put max pools:
            convs = int(params[self.hyper_map['num_conv_layers']])
            pools = int(params[self.hyper_map['num_maxpools']])
            add_max_pool_after_layer = np.arange(1, convs+1)[::-1][0:pools]
            
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
            if int(params[self.hyper_map['num_conv_layers']]) > 1:
                
                for l in range(1, int(params[self.hyper_map['num_conv_layers']])):
                    
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

            print("Cross validating..")
            scores = []

            for i in range(n):

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

            return scores


        # run the experiment nfolds times and print mean and std.
        scores = cross_validate(self.x_train, self.y_train, self.x_test, self.y_test, params, self.nfolds)
        print("Scores: {}.\nMean: {}%. Standard deviation: {}%".format(scores, round(np.mean(scores)*100, 2), round(np.std(scores)*100, 2)))
        
        # log after every function call / for every set of parameters
        if self.log:
            self.param_log = pd.concat([self.param_log, pd.DataFrame(np.reshape(params, (1,self.dim)), columns=self.hyperparams)])
            self.scores_log = pd.concat([self.scores_log, pd.DataFrame(np.reshape(scores, (1,self.nfolds)), columns=np.arange(1,self.nfolds+1))])
            self.param_log.to_csv("./Results/params_log.csv", index=False)
            self.scores_log.to_csv("./Results/scores_log.csv", index=False)

        # Return negative mean value for pySOT to minimize
        return -np.mean(scores)



class OneShotTransferCNN():
    """ Class used for one shot image classification with Transfer Learning. 

    :param log: Boolean that indicates whether to write the results to a csv file.
    :param folds: Number of experiments to run for each combination of hyperparameter values.
    :param verbose: Whether to print detailed information to the console.
    """

    def __init__(self, log=False, folds=10, batchsize=128, verbose=True):
        
        # specify the parameter ranges as [min, max].
        # first continuous, then integer params.
        self.rgs = {'learning_rate': [0.0001, 0.01],
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
                    'finetune_epochs': [50, 1000],
                    'num_fixed_layers': [0, 8]}

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


        def define_model(params):
            """ Creates the Keras model based on given parameters. 

            :param params: The parameters of the object indicating the architecture of the NN.
            :returns: A tuple of the model and datagenerator. 
            """

            # determine where to put max pools:
            convs = int(params[self.hyper_map['num_conv_layers']])
            pools = int(params[self.hyper_map['num_maxpools']])
            add_max_pool_after_layer = np.arange(1, convs+1)[::-1][0:pools]
            
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
            if int(params[self.hyper_map['num_conv_layers']]) > 1:
                
                for l in range(1, int(params[self.hyper_map['num_conv_layers']])):
                    
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

            print("Cross validating..")
            scores = []

            for i in range(n):

                # get data
                x_target_labeled, y_target, x_test, y_test, _, x_auxiliary, y_auxiliary = \
                    split_and_select_random_data(x, y, xtest, ytest, num_target_classes=5, num_examples_per_class=1)

                # define model according to parameters
                model, datagen = define_model(params)
                
                # fit the model on batches with real-time data augmentation:
                print("fit {}:".format(i+1))
                model.fit_generator(datagen.flow(x_auxiliary, y_auxiliary, batch_size=self.batchsize),
                                                 steps_per_epoch=int(round(x_auxiliary.shape[0]/self.batchsize, 0)),
                                                 epochs=params[self.hyper_map['epochs']], verbose=0)

                # fix layers
                for layer in model.layers[0:params[self.hyper_map['num_fixed_layers']]]:
                    layer.trainable = False

                # fine tune the model
                finetune_datagen = ImageDataGenerator(width_shift_range=params[self.hyper_map['finetune_width_shift']],
                                                      height_shift_range=params[self.hyper_map['finetune_height_shift']],
                                                      shear_range=params[self.hyper_map['finetune_shear']],
                                                      zoom_range=params[self.hyper_map['finetune_zoom']],
                                                      horizontal_flip=params[self.hyper_map['finetune_horizontal_flip']],
                                                      rotation_range=params[self.hyper_map['finetune_rotation']])
                
                model.fit_generator(finetune_datagen.flow(x_target_labeled, y_target, batchsize=x_target_labeled.shape[0],
                                                          steps_per_epoch=1, epochs=params[self.hyper_map['finetune_epochs']],
                                                          verbose=0))


                # evaluate on test set
                loss, accuracy = model.evaluate(x_test, y_test, verbose=0, batch_size=y.shape[0])

                # report score
                print("test accuracy: {}%.".format(round(accuracy*100, 2)))
                scores.append(accuracy)

            return scores


        # run the experiment nfolds times and print mean and std.
        scores = cross_validate(self.x_train, self.y_train, self.x_test, self.y_test, params, self.nfolds)
        print("Scores: {}.\nMean: {}%. Standard deviation: {}%".format(scores, round(np.mean(scores)*100, 2), round(np.std(scores)*100, 2)))
        
        # log after every function call / for every set of parameters
        if self.log:
            self.param_log = pd.concat([self.param_log, pd.DataFrame(np.reshape(params, (1,self.dim)), columns=self.hyperparams)])
            self.scores_log = pd.concat([self.scores_log, pd.DataFrame(np.reshape(scores, (1,self.nfolds)), columns=np.arange(1,self.nfolds+1))])
            self.param_log.to_csv("./Results/params_log.csv", index=False)
            self.scores_log.to_csv("./Results/scores_log.csv", index=False)

        # Return negative mean value for pySOT to minimize
        return -np.mean(scores)