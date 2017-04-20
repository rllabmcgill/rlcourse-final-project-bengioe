#from __future__ import print_function
from sklearn.model_selection import StratifiedKFold as SKF #for balanced minibatch option
import cPickle
import numpy
import numpy as np
import gzip
from time import gmtime, strftime
import pickle
import sys
import os
from operator import itemgetter

svhn_path = None
for path in ['/home/2014/ebengi/a10/data/svhn_crop/svhn_shuffled_trainX.raw',
             '/data/svhn_crop/svhn_shuffled.pkl',
             '/your/path/here',
             './svhn_shuffled.pkl']:
    if os.path.exists(path):
        svhn_path = path
        break
if svhn_path is None:
    print('>>> Warning <<< couldnt find SVHN')

import os
_proc_status = '/proc/%d/status' % os.getpid()

_scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
          'KB': 1024.0, 'MB': 1024.0*1024.0}

def _VmB(VmKey):
    '''Private.
    '''
    global _proc_status, _scale
     # get pseudo file  /proc/<pid>/status
    try:
        t = open(_proc_status)
        v = t.read()
        t.close()
    except:
        return 0.0  # non-Linux?
     # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
     # convert Vm value to bytes
    return float(v[1]) * _scale[v[2]]


def memory(since=0.0):
    '''Return memory usage in bytes.
    '''
    return _VmB('VmSize:') - since


def resident(since=0.0):
    '''Return resident memory usage in bytes.
    '''
    return _VmB('VmRSS:') - since


class SVHN:
    def __init__(self, flat=True):
        path = svhn_path
        print(resident())
        if path.endswith('.pkl'):
            train,test = cPickle.load(open(svhn_path,'r'))
            train[1] = train[1].flatten() - 1
            test[1] = test[1].flatten() - 1
        elif path.endswith('.raw'):
            path = path[:-len('_trainX.raw')]
            print('raw',path)
            print('resident:',resident())
            train = [np.memmap(path+'_trainX.raw',mode='r',shape=(604388, 32, 32, 3)),
                     np.memmap(path+'_trainY.raw',mode='r',shape=(604388,))]
            test = [np.memmap(path+'_testX.raw',mode='r',shape=(26032, 32, 32, 3)),
                    np.memmap(path+'_testY.raw',mode='r',shape=(26032,))]
            print('resident:',resident())
        n = 580000
        if flat:
            train[0] = train[0].reshape((train[0].shape[0],-1))
            test[0] = test[0].reshape((test[0].shape[0],-1))
        self.train = [train[0][:n], train[1][:n]]
        self.valid = [train[0][n:], train[1][n:]]
        self.test = test
        print('resident:',resident())

    def runEpoch(self, dataGenerator, func):
        n = dataGenerator.next()
        #print n
        stats = None
        e=-1
        print('\n')
        print "minibatch",
        for a in dataGenerator:
            e += 1
            if e %10 == 0 :
                print '%i,'%e,
#                sys.stdout.flush()
            s = func(*a)
            if stats is None:
                stats = map(np.float32,s)
            else:
                for i,j in enumerate(s):
                    stats[i] += j
        #print stats, [i/n for i in stats]
        return [i / n for i in stats]

    def validMinibatches(self, mbsize=32):
        return self.minibatches(self.valid, mbsize, True)
    
    def trainMinibatches(self, mbsize=32,balanced=True):
        if balanced :
            return self.balancedMinibatches(self.train, mbsize, True)
        else :
            return self.minibatches(self.train, mbsize, True)

    def balancedMinibatches(self, dset, mbsize=32, yieldN=False):
        n = dset[0].shape[0]
        if False :
            if yieldN:
                yield n
            nb_minibatch = n / mbsize + bool(n % mbsize)
            skf = SKF(n_splits=nb_minibatch,shuffle=True)
            print('starting split %s'%strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            split = map(itemgetter(1),skf.split(dset[0],dset[1]))
            print('split done%s'%strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            f = open('./split.pkl','wb')
            split = list(split)
            print('saving')
            pickle.dump(split,f)
            print('finished')
            f.close()

        else :
            f = open('./split.pkl', 'rb')
            split = pickle.load(f)
            f.close()
            print('split loaded')

        for idx in split:
            idx = idx
            yield numpy.float32(dset[0][idx] / 255.), dset[1][idx]

    def minibatches(self, dset, mbsize=32, yieldN=False):
        n = dset[0].shape[0]
        indexes = np.arange(n)
        np.random.shuffle(indexes)
        if yieldN:
            yield n
        for i in range(n/mbsize + bool(n%mbsize)):
            idx = indexes[i*mbsize:(i+1)*mbsize]
            yield numpy.float32(dset[0][idx]/255.), dset[1][idx]
