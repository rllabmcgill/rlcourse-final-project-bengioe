import cPickle
import numpy
import numpy as np
import gzip
import os


for path in ['/data/svhn_crop/svhn_shuffled.pkl',
              '/your/path/here']:
    if os.path.exists(path):
        svhn_path = path
        break



class SVHN:
    def __init__(self, flat=True):
        train,test = cPickle.load(open(svhn_path,'r'))
        train[1] = train[1].flatten() - 1
        test[1] = test[1].flatten() - 1
        n = 580000
        if flat:
            train[0] = train[0].reshape((train[0].shape[0],-1))
            test[0] = test[0].reshape((test[0].shape[0],-1))
        self.train = [train[0][:n], train[1][:n]]
        self.valid = [train[0][n:], train[1][n:]]
        self.test = test
        
    def runEpoch(self, dataGenerator, func):
        n = dataGenerator.next()
        stats = None
        for a in dataGenerator:
            s = func(*a)
            if stats is None:
                stats = map(np.float32,s)
            else:
                for i,j in zip(stats,s):
                    i += j
        return [i / n for i in stats]

    def validMinibatches(self, mbsize=32):
        return self.minibatches(self.valid, mbsize)
    
    def trainMinibatches(self, mbsize=32):
        return self.minibatches(self.train, mbsize)

    def minibatches(self, dset, mbsize=32):
        n = dset[0].shape[0]
        indexes = np.arange(n)
        np.random.shuffle(indexes)
        yield n
        for i in range(n/mbsize + bool(n%mbsize)):
            idx = indexes[i*mbsize:(i+1)*mbsize]
            yield dset[0][idx], dset[1][idx]
