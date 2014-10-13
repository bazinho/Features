import os,sys
import cPickle
import gzip
import numpy

import matplotlib.pyplot as plt

class dataset:
  """Class that creates a malware dataset"""
  
  def __init__(self, classfile, featuredir = ".", outputfile = "dataset.pkl.gz", verbose = True):
    self.classfile = classfile
    self.featuredir = featuredir
    self.outputfile = outputfile
    self.verbose = verbose
    self.mapFile = {}
    self.mapClassToId = {}    
    self.readClassfile()
    self.numfeatures = 784
    #trainset_size = dataset_size * 60 / 100
    #validset_size = dataset_size * 20 / 100
    #testset_size = dataset_size - trainset_size - validset_size
    #trainset = numpy.zeros((trainset_size,numfeatures), dtype=numpy.float32) , numpy.zeros(trainset_size, dtype=numpy.int32)
    #validset = numpy.zeros((validset_size,numfeatures), dtype=numpy.float32) , numpy.zeros(validset_size, dtype=numpy.int32)
    #testset  = numpy.zeros((testset_size,numfeatures),  dtype=numpy.float32) , numpy.zeros(testset_size,  dtype=numpy.int32)
    #self.dataset = trainset, validset, testset
    self.createGZIP()
        
  def readClassfile(self):
    if os.path.isfile(self.classfile):
      with open(self.classfile, 'rb') as fdclassfile:
	self.mapClassToId['Benign'] = len(self.mapClassToId) + 1
	for line in fdclassfile:
	  if line[0] != '#':	# ignore comments
	    filename, classname = line.split()[0].split(",")
	    self.mapFile[filename] = {'class': classname}
	    if classname not in self.mapClassToId:
	      self.mapClassToId[classname] = len(self.mapClassToId) + 1
	    if self.verbose:
	      print("File: %s - Class: %s (%d)" %(filename,classname,self.mapClassToId[classname])) 

  def extractFileContent(self,filename):
    if os.path.isfile(os.path.join(self.featuredir,filename)):
      with open(os.path.join(self.featuredir,filename), 'rb') as fdfeaturefile:
	if filename not in self.mapFile:
	  self.mapFile[filename] = {'class': 'Benign'}
	self.content = numpy.array(bytearray(fdfeaturefile.read()))
        
  def createGZIP(self):
    self.M = 0
    for (dirpath, dirnames, filelist) in os.walk(self.featuredir):
      for filename in filelist:
        self.M += os.path.getsize(os.path.join(dirpath,filename))-self.numfeatures+1
    if self.verbose:
      print("M: %d" %(self.M)) 
    self.data  = numpy.zeros((self.M, self.numfeatures), dtype=numpy.float32)
    self.label = numpy.zeros((self.M), dtype=numpy.int32)
    
    m = 0
    for (dirpath, dirnames, filelist) in os.walk(self.featuredir):
      for filename in filelist:
        self.extractFileContent(filename)
        for w in xrange(0,len(self.content)-self.numfeatures+1):
          self.data[m]  += self.content[w:w+self.numfeatures]
          self.label[m] += self.mapClassToId[self.mapFile[filename]['class']]
          m += 1
    self.dataset = (self.data, self.label), (self.data, self.label), (self.data, self.label)
    #print "Dataset:"
    #print self.dataset
    #print "M x N: ", self.dataset[0].shape
    #plt.hist(self.data, bins = range(self.data.max()))
    #plt.show()
    fdout = gzip.open(self.outputfile, 'wb')
    #fdout = open(self.outputfile, 'wb')
    cPickle.dump(self.dataset, fdout)
    fdout.close()
    
  def showClasses(self):
    for c in sorted(self.mapClassToId, key=self.mapClassToId.get):
      print self.mapClassToId[c], ":", c
