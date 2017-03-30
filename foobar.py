import numpy
import numpy as np

import theano
import theano.tensor as T

from util import make_param, srng, sgd, SVHN

import shelve

def randargmax(b,**kw):
    """ a random tie-breaking argmax"""
    return np.argmax(np.random.random(b.shape) * (b==b.max()),**kw)

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
        self.visits = np.zeros((sum(nhid), npart)) + 1e-3
        
    def makePartition(self):
        # greedy partition
        return randargmax(self.total_rewards / self.visits, axis=1)

    def partitionFeedback(self, partition, reward):
        self.total_rewards[np.ogrid[:self.n], partition] += reward
        self.visits[np.ogrid[:self.n], partition] += 1


class ReinforceComputationPolicy:
    def __init__(self, npart, nin):
        self.W = make_param((nin, npart))
        self.b = make_param((npart,))
        self.params = [self.W, self.b]

    def applyAndGetFeedbackMethod(self, x):
        probs = T.nnet.sigmoid(T.dot(x,self.W)+self.b) * 0.98 + 0.01
        mask = srng.uniform(probs.shape) < probs
            
        return mask, probs, self.rewardFeedback(probs, mask)
    def rewardFeedback(self, probs, mask):
        def f(reward, lr):
            loss = theano.gradient.disconnected_grad(-reward)
            reinf = T.mean(T.mean(T.log(probs * mask + (1-probs) * (1-mask)),axis=1) * loss)
            grads = T.grad(reinf, self.params)
            updates = sgd(self.params, grads, lr)
        return f


class LazyNet:
    def __init__(self, npart, lr):
        self.target = TargetNet()
        self.partitionner = BanditPartitionner(npart, self.target.nhid)
        self.comppol = ReinforceComputationPolicy(npart, self.target.nin)
        self.lr = lr
        
    def performUpdate(self, dataset, maxEpochs=50):
        print 'creating theano graph...'
        x = T.matrix()
        y = T.ivector()
        partition = self.partitionner.makePartition()
        partitionMask, probs, policyFeedbackMethod = self.comppol.applyAndGetFeedbackMethod(x)
        masks = [partition[start:end]
                 for start,end in zip(np.cumsum(self.target.nhid) - self.target.nhid[0],
                                      np.cumsum(self.target.nhid))]
        
        o = self.target.applyToXWithMask(x, masks)

        mbloss = T.nnet.categorical_crossentropy(o,y)
        updates = policyFeedbackMethod(mbloss, self.lr)
        acc = T.sum(T.eq(T.argmax(o,axis=1),y))
        loss = T.sum(mbloss)
        print 'compiling'
        learn = theano.function([x,y],[loss, acc],updates=updates)
        test = theano.function([x,y],[loss, acc, T.mean(probs, axis=0)])

        print 'training computation policy'
        tolerance = 5
        last_validation_loss = 1
        for epoch in range(maxEpochs):
            train_loss, train_acc = dataset.runEpoch(dataset.trainMinibatches(), learn)
            valid_loss, valid_acc, probs = dataset.runEpoch(dataset.validMinibatches(), test)
            print epoch, train_loss, train_acc, valid_loss, valid_acc
            print probs
            if valid_loss > last_validation_loss:
                tolerance -= 1
            last_validation_loss = valid_loss

svhn = SVHN()
net = LazyNet(4, 0.0001)
net.performUpdate(svhn)
