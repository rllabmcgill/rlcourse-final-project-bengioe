import inspect
import traceback
import numpy

from collections import OrderedDict

import theano
import theano.tensor as T
from theano.tensor.nnet.abstract_conv import AbstractConv2d_gradInputs
from theano.sandbox.rng_mrg import MRG_RandomStreams


def nprand(shape, k):
    return numpy.float32(numpy.random.uniform(-k,k, shape))

def make_param(shape):
    if len(shape) == 1:
        return theano.shared(nprand(shape,0),'b')
    elif len(shape) == 2:
        return theano.shared(nprand(shape, numpy.sqrt(6./sum(shape))), 'W')
    elif len(shape) == 4:
        return theano.shared(nprand(shape, numpy.sqrt(6./(shape[1]+shape[0]*numpy.prod(shape[2:])))), 'W')
    raise ValueError(shape)

def sgd(params, grads, lr):
    return [(i, i-lr*gi) for i,gi in zip(params,grads)]

def _log(*args):
    if _log.on:
        print " ".join(map(str,args))
_log.on = False

srng = MRG_RandomStreams(seed=142)
