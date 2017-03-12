import numpy
import pylab #for graphing
import pickle
import sys

data = {}
modelName = "A3stats.pickle"
if len(sys.argv) > 1:
	modelName = sys.argv[1]
with open('%s'%modelName) as infile:
	if '.pickle' in modelName:
		data = pickle.load(infile)
'''
for key in data.keys():
	data[key] = numpy.array(data[key],dtype='float32')
'''
pylab.plot(data['epoch'],data['tr_error'], '--go',label='Training Error')
pylab.plot(data['epoch'],data['tr_accuracy'],'--ko',label='Training Accuracy')
pylab.plot(data['epoch'],data['v_error'], '-ro',label='Validation Error')
pylab.plot(data['epoch'],data['v_accuracy'],'-bo',label='Validation Accuracy')
pylab.xlabel("Epoch")
pylab.ylabel("Cross entropy error")
pylab.ylim(0,max(data['error']) if max(data['error']) < 20000 else 20000)
pylab.title(modelName)
pylab.legend(loc='upper right')
#pylab.savefig('.png'%modelName)
pylab.show()#enter param False if running in iterative mode