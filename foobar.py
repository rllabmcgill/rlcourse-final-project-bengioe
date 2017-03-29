import numpy
import numpy as np

import theano
import theano.tensor as T

from util import make_param, srng

import shelve


class TargetNet:
    def __init__(self):
        params = shelve.open('./svhn_mlp/params.db')

        print params.keys()
        print [params[i].shape for i in params]

        self.params = map(theano.shared, [params[i] for i in
                                          ['Wt0_0', 'bt0_1', 'Wt1_2', 'bt1_3', 'Woutput_4', 'boutput_5']])
        self.nhid = [self.params[i*2].get_value().shape[1] for i in range(2)]
        self.nin = self.params[0].get_value().shape[0]
        
    def applyToXWithMask(self, x, masks):
        o = x
        for i,m in enumerate(masks):
            W,b = self.params[i*2:i*2+2]
            o = T.nnet.relu(T.dot(o,W)+b) * m
        i += 1
        W,b = self.params[i*2:i*2+2]
        o = T.nnet.softmax(T.dot(o,W)+b)
        return o

class BanditPartitionner:
    def __init__(self, npart, nhid):
        self.n = sum(nhid)
        self.total_rewards = np.zeros((sum(nhid), npart))
        self.visits = np.zeros((sum(nhid), npart))
        
    def makePartition(self):
        # greedy partition
        return np.argmax(self.total_rewards / self.visits, axis=1)

    def partitionFeedback(self, partition, reward):
        self.total_rewards[np.ogrid[:self.n], partition] += reward
        self.visits[np.ogrid[:self.n], partition] += 1


class ReinforceComputationPolicy:
    def __init__(self, npart, nin):
        self.W = make_param((nin, npart))
        self.b = make_param((npart,))
        self.params = [self.W, self.b]

    def applyAndGetFeedbackMethod(self, x):
        probs = T.nnet.sigmoid(T.dot(x,W)+b) * 0.98 + 0.01
        mask = srng.uniform(probs.shape) < probs
            
        return mask, self.rewardFeedback(probs, mask)
    def rewardFeedback(self, probs, mask):
        def f(reward, lr):
            loss = theano.gradient.disconnected_grad(-reward)
            grads = T.grad(T.mean(T.log(probs) * loss), self.params)
            updates = sgd(self.params, self.grads, lr)
        return f


class LazyNet:
    def __init__(self, npart):
        self.target = TargetNet()
        self.partitionner = BanditPartitionner(npart, self.target.nhid)
        self.comppol = ReinforceComputationPolicy(npart, self.target.nin)

    def performUpdate(self, xtrain, ytrain, xvalid, yvalid):
        print 'creating theano graph...'
        x = T.matrix()
        y = T.ivector()
        partition = self.partitionner.makePartition()
        mask, policyFeedbackMethod = self.comppol.applyAndGetFeedbackMethod(x)
        o = self.target.applyToXWithMask(x, masks)
