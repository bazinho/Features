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
    self.createPKL()
        
  def readClassfile(self):
    """ 
    Creates two dictionaries:
	self.mapFile[filename] = {'class': classname} 
	self.ClassToId[classname] = classId
    """
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
    """ 
    Creates self.contents: a numpy.array with file contents 
    """
    if os.path.isfile(os.path.join(self.featuredir,filename)):
      with open(os.path.join(self.featuredir,filename), 'rb') as fdfeaturefile:
	if filename not in self.mapFile:
	  self.mapFile[filename] = {'class': 'Benign'}
	self.content = numpy.array(bytearray(fdfeaturefile.read()))
        
  def createPKL(self):
    """
    Creates pickle file (PKL) containing datasets
    """
    # Set M as the sum of all malware file sizes in featuredir 
    self.M = 0
    for (dirpath, dirnames, filelist) in os.walk(self.featuredir):
      for filename in filelist:
        #self.M += os.path.getsize(os.path.join(dirpath,filename))-self.numfeatures+1
        self.M += os.path.getsize(os.path.join(dirpath,filename))/self.numfeatures
    if self.verbose:
      print("M: %d" %(self.M))
      
    # Creates M x N data array and M label vector
    self.data  = numpy.zeros((self.M, self.numfeatures), dtype=numpy.float32)
    self.label = numpy.zeros((self.M), dtype=numpy.int32)
    
    # For each malware in featuredir extracts nonoverlap windows of 
    # file content and insert into dataset with malware label
    m = 0
    for (dirpath, dirnames, filelist) in os.walk(self.featuredir):
      for filename in filelist:
        self.extractFileContent(filename)
        #for w in xrange(0,len(self.content)-self.numfeatures+1):
          #self.data[m]  += self.content[w:w+self.numfeatures]
          #self.label[m] += self.mapClassToId[self.mapFile[filename]['class']]
          #m += 1
        for w in xrange(0,len(self.content)/self.numfeatures):
          self.data[m]  += self.content[w*self.numfeatures:w*self.numfeatures+self.numfeatures]
          self.label[m] += self.mapClassToId[self.mapFile[filename]['class']]
          m += 1
          
    # Splits dataset randomly into trainset (60%), validset (20%) and testset (20%)
    trainset_size = M * 0.6
    validset_size = M * 0.2
    testset_size = M - trainset_size - validset_size
    randM = numpy.random.permutation(M)
    self.dataset = (self.data[randM[0:trainset_size]], self.label[randM[0:trainset_size]]),
                   (self.data[randM[trainset_size:trainset_size+validset_size]], self.label[randM[trainset_size:trainset_size+validset_size]]),
                   (self.data[randM[trainset_size+validset_size:]], self.label[randM[trainset_size+validset_size:]])
    if self.verbose:
      print "Datasets:\n",self.dataset
      print "Trainset size: ", self.dataset[0].shape
      print "Validset size: ", self.dataset[1].shape
      print "Testset size: ", self.dataset[2].shape
      
    # Creates pickle file containing dataset 
    fdout = open(self.outputfile, 'wb')
    cPickle.dump(self.dataset, fdout)
    fdout.close()
    
  def showClasses(self):
    for c in sorted(self.mapClassToId, key=self.mapClassToId.get):
      print self.mapClassToId[c], ":", c
