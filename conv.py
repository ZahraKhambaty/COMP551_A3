import numpy
import scipy.io as sio
import theano
import lasagne
import theano.tensor as T
import os, sys

from collections import OrderedDict
import pylab #for graphing

from random import shuffle

import time
import math
from sklearn.model_selection import train_test_split

def createNetwork(dimensions, input_var):
	#dimensions = (1,1,data.shape[0],data.shape[1]) #We have to specify the input size because of the dense layer
	#We have to specify the input size because of the dense layer
	print ("Creating Network...")

	print ('Input Layer:')
	network = lasagne.layers.InputLayer(shape=dimensions,input_var=input_var)
	
	print '	',lasagne.layers.get_output_shape(network)
	print ('Hidden Layer:')

	#network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(5,5), pad ='same',nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(5,5), pad ='same',nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2, 2))
	print '	',lasagne.layers.get_output_shape(network)
	
	#network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(3,3), pad='same',nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.Conv2DLayer(network, num_filters=256, filter_size=(3,3), pad='same',nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2, 2))
	print '	',lasagne.layers.get_output_shape(network)
	
	#network = lasagne.layers.Conv2DLayer(network, num_filters=256, filter_size=(3,3), pad='same',nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.Conv2DLayer(network, num_filters=256, filter_size=(3,3), pad='same',nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2, 2))
	print '	',lasagne.layers.get_output_shape(network)
	'''
	network = lasagne.layers.DenseLayer(network,num_units=2048,nonlinearity=lasagne.nonlinearities.rectify)
	print '	',lasagne.layers.get_output_shape(network)

	network = lasagne.layers.DenseLayer(network,num_units=1024,nonlinearity=lasagne.nonlinearities.rectify)
	print '	',lasagne.layers.get_output_shape(network)
	'''
	network = lasagne.layers.DenseLayer(network, num_units=40, nonlinearity = lasagne.nonlinearities.softmax)
	print ('Output Layer:')
	print '	',lasagne.layers.get_output_shape(network)


	return network

def createTrainer(network,input_var,y):
	print ("Creating Trainer...")
	#output of network
	out = lasagne.layers.get_output(network)
	#get all parameters from network
	params = lasagne.layers.get_all_params(network, trainable=True)
	#calculate a loss function which has to be a scalar
	cost = T.nnet.categorical_crossentropy(out, y).mean()
	#calculate updates using ADAM optimization gradient descent
	updates = lasagne.updates.adam(cost, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
	#theano function to compare brain to their masks with ADAM optimization
	train_function = theano.function([input_var, y], updates=updates) # omitted (, allow_input_downcast=True)
	return train_function

def createValidator(network, input_var, y):
	print ("Creating Validator...")
	#We will use this for validation
	testPrediction = lasagne.layers.get_output(network, deterministic=True)			#create prediction
	testLoss = lasagne.objectives.categorical_crossentropy(testPrediction,y).mean()   #check how much error in prediction
	testAcc = T.mean(T.eq(T.argmax(testPrediction, axis=1), T.argmax(y, axis=1)),dtype=theano.config.floatX)	#check the accuracy of the prediction

	validateFn = theano.function([input_var, y], [testLoss, testAcc])	 #check for error and accuracy percentage
	return validateFn

def saveModel(network,saveLocation='',modelName='brain1'):

	networkName = '%s%s.npz'%(saveLocation,modelName)
	print ('Saving model as %s'%networkName)
	numpy.savez(networkName, *lasagne.layers.get_all_param_values(network))

def loadModel(network, model='brain1.npz'):

	with numpy.load(model) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))] #gets all param values
		lasagne.layers.set_all_param_values(network, param_values)		  #sets all param values
	return network


def main():

	#testReserve = 0.1
	#validationReserve = 0.2
	#trainingReserve = 1-(testReserve+validationReserve)
	input_var = T.tensor4('input')
	y = T.dmatrix('truth')

	trainFromScratch = True
	#epochs = 10
	samplesperEpoch = 100 # how many divisions there are in the data when training since it all can't fit in memory

	#MODIFY THESE
	#*******************************************
	trainTime = 1 #in hours
	modelName='A3'
	#*******************************************


	trainX = numpy.load('data/tinyX.npy') # this should have shape (26344, 3, 64, 64)
	trainY = numpy.load('data/tinyY.npy') 
	print ("%i samples found"%trainX.shape[0])
	#Set up truth values

	new_y = numpy.array([numpy.zeros(40)],dtype='float32')
	for categ in trainY:
		temp = numpy.array([numpy.zeros(40)],dtype='float32')
		numpy.put(temp,categ,1)
		new_y = numpy.concatenate((new_y,temp),axis=0)
	new_y = new_y[1:]
	
	#This reserves the correct amount of samples for training, testing and validating
	#trainingSet = trainX[:int(trainingReserve*trainX.shape[0])]
	#testSet = trainX[int(trainingReserve*trainX.shape[0]):-int(testReserve*trainX.shape[0])]
	#validationSet = trainX[int(testReserve*trainX.shape[0] + int(trainingReserve*trainX.shape[0])):]

	#trainingLabel = new_y[:int(trainingReserve*new_y.shape[0])]
	#testLabel = new_y[int(trainingReserve*new_y.shape[0]):-int(testReserve*new_y.shape[0])]
	#validationLabel = new_y[int(testReserve*new_y.shape[0] + int(trainingReserve*new_y.shape[0])):]
	
	trainingSet, validationSet, trainingLabel, validationLabel = train_test_split(trainX,new_y, test_size=0.3, random_state=42)

	networkDimensions = (None,trainingSet.shape[1],trainingSet.shape[2],trainingSet.shape[3])
	network  = createNetwork(networkDimensions, input_var)
	trainer = createTrainer(network,input_var,y)
	validator = createValidator(network,input_var,y)

	if not trainFromScratch:
		print ('loading a previously trained model...\n')
		network = loadModel(network,'A3.npz')


	#print ("Training for %s epochs with %s samples per epoch"%(epochs,samplesperEpoch))
	record = OrderedDict(epoch=[],tr_error=[],tr_accuracy=[],v_error=[],v_accuracy=[])
	print ("Training for %s hour(s) with %s samples per epoch"%(trainTime,samplesperEpoch))
	epoch = 0
	startTime = time.time()
	timeElapsed = time.time() - startTime

	#for epoch in xrange(epochs):            #use for epoch training
	while (time.time()-startTime)/3600 < trainTime :     #use for time training
		epochTime = time.time()
		print ("--> Epoch: %d | Time left: %.2f hour(s)"%(epoch,trainTime-(time.time()-startTime)/3600))
		chunk = int(math.ceil(trainingSet.shape[0]/samplesperEpoch))
		for i in xrange(samplesperEpoch):

			data = trainingSet[chunk*i:chunk*i + chunk]
			label = trainingLabel[chunk*i:chunk*i + chunk]
			trainer(data, label)
			timeElapsed = time.time() - epochTime
			percentage = float(i+1) / float(samplesperEpoch) * 100
			sys.stdout.flush()
			sys.stdout.write("\r %d training steps complete: %.2f%% done epoch in %ds" % (i + 1, percentage, timeElapsed))

		tr_error, tr_accuracy = validator(trainingSet[:4000], trainingLabel[:4000])
		v_error, v_accuracy = validator(validationSet[:4000], validationLabel[:4000])			     #pass modified data through network
		record['tr_error'].append(tr_error); record['tr_accuracy'].append(tr_accuracy)
		record['v_error'].append(v_error); record['v_accuracy'].append(v_accuracy)
		record['epoch'].append(epoch)
		epochTime = time.time() - epochTime
		print ("\n	error: %s and accuracy: %s in %.2fs\n"%(v_error,v_accuracy,epochTime))
		epoch+=1

	#validateNetwork(network,input_var,testSet)

	saveModel(network=network,modelName=modelName)
	#import pudb; pu.db
	#save metrics to pickle file to be opened later and displayed
	import pickle

	with open('%sstats.pickle'%modelName,'w') as output:

		pickle.dump(record,output)

	import pudb; pu.db
	
if __name__ == "__main__":
    main()