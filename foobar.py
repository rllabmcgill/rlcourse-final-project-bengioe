import numpy
import numpy as np

import theano
import theano.tensor as T
from theano.tests.breakpoint import PdbBreakpoint

from util import make_param, srng, sgd, SVHN

import shelve
import cPickle as pickle

from itertools import product

def randargmax(b,**kw):
    """ a random tie-breaking argmax"""
    return np.argmax(np.random.random(b.shape) * (b==b.max()),**kw)

class TargetNet:
    def __init__(self, reloadFrom=None, architecture=None):
        self.params = []
        if reloadFrom is not None:
            print 'reloading from',reloadFrom
            if reloadFrom.endswith('.db'):
                params = shelve.open(reloadFrom)
                print params.keys()
                print [params[i].shape for i in params]

                self.params = map(theano.shared, [params[i] for i in
                                                  ['Wt0_0', 'bt0_1', 'Wt1_2', 'bt1_3', 'Woutput_4', 'boutput_5']])
            else:
                self.params = map(theano.shared, pickle.load(open(reloadFrom)))
                
            self.nhid = [self.params[i*2].get_value().shape[1] for i in range(2)]
            self.nin = self.params[0].get_value().shape[0]
            self.nout = self.params[-2].get_value().shape[1]


        elif architecture is not None:
            print 'making net from scratch',architecture
            for inp,out in zip(architecture[:-1],architecture[1:]):
                self.params.append(make_param((inp,out)))
                self.params.append(make_param((out,)))
                print (inp,out)
            self.nhid = architecture[:-1]
            self.nin = architecture[0]
            self.nout=architecture[-1]
        else:
            raise ValueError()
        
    def applyToXWithMask(self, x, masks):
        o = x
        for i,m in enumerate(masks):
            W,b = self.params[i*2:i*2+2]
            o = T.nnet.relu(T.dot(o,W)+b) * m
            #o,m = PdbBreakpoint('test')(1, o, T.as_tensor_variable(m))
        i += 1
        W,b = self.params[i*2:i*2+2]
        o = T.nnet.softmax(T.dot(o,W)+b)
        return o

    def applyToX(self, x, dropout=None,return_activation=False):
        o = x
        if return_activation:   hs = [o]

        for i in range(len(self.params)/2-1):
            W,b = self.params[i*2:i*2+2]
            o = T.nnet.relu(T.dot(o,W)+b)
            if dropout is not None:
                o = o * (srng.uniform(o.shape) < dropout)
            if return_activation:   hs.append(o)

        i += 1
        W,b = self.params[i*2:i*2+2]
        o = T.nnet.softmax(T.dot(o,W)+b)

        if return_activation:  return o,hs

        return o

    def get_weights(self):
        return [self.params[2 * i] for i in range(len(self.nhid))]


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

class ContextualBanditPartitionner:
    """ Linear Response Banding Algorithm. We use the weights of connections to the neuron as context. We consider that each layer is a different bandit problem.  """
    def __init__(self, npart, nhid, net):
        self.n = sum(nhid)
        self.nhid = nhid
        self.npart = npart
        self.nlayers = len(nhid)
        self.weights = list(map(lambda x: x.get_value(), net.get_weights()))

        self.total_rewards = np.zeros((sum(nhid), npart))
        self.visits = np.zeros((sum(nhid), npart)) + 1e-3

#        self.exploit_total_rewards = np.zeros((sum(nhid), npart))
#        self.exploit_visits = np.zeros((sum(nhid), npart)) + 1e-3
#        self.explore_total_rewards = np.zeros((sum(nhid), npart))
#        self.explore_visits = np.zeros((sum(nhid), npart)) + 1e-3

        self.beta_h = [np.zeros((npart, nh)) for nh in nhid]
#        self.beta_t = [np.zeros((npart, nh)) for nh in nhid]

    def makePartition(self):

        X = self.weights
        part = np.zeros((self.n,))
        i = 0
        for l in range(self.nlayers):
            for n in range(self.nhid[l]):
                if False :
                # if gap between action is big enough use beta tilde instead of beta hat (whatever that means)
#                if np.min([np.abs((beta_t[l][a1] - beta_t[l][a2]).T *X[l][n]) for a1,a2 in .....]) < h / 2.0:
                    part[i] = randargmax(np.array([(self.beta_t[l][a]).T * X[l][n] for a in range(self.npart)]))
                else:
                    part[i] = randargmax(np.array([(self.beta_h[l][a]).T * X[l][n] for a in range(self.npart)]))
            i += 1
        return part

    def betaUpdate(self):
        X = self.weights
        y = 0
        for l, a in product(range(self.nlayers), range(self.npart)):
            yy = y + self.nhid[l]
            coeff, R = X[l], self.total_rewards[y:yy] / self.visits[y:yy]
            self.beta_h[l][a] = np.linalg.ltsqr(coeff, R)

            # only on exploration round
#            self.beta_h[l][a] = least_sqr(np.hstack([explore_coeff, exploit_coeff]), np.vstack([explore_R, exploit_R]))
#            self.beta_h[l][a] = least_sqr(explore_coeff, explore_R)
            y = yy

    def partitionFeedback(self, partition, reward):
        self.total_rewards[np.ogrid[:self.n], partition] += reward
        self.visits[np.ogrid[:self.n], partition] += 1
        self.betaUpdate()



class ReinforceComputationPolicy:
    def __init__(self, npart, nin):
        self.W = make_param((nin, npart))
        self.b = make_param((npart,))
        self.params = [self.W, self.b]

    def applyAndGetFeedbackMethod(self, x):
        probs = T.nnet.sigmoid(T.dot(x,self.W)+self.b) * 0.95 + 0.025
        #probs = theano.printing.Print('probs')(probs)
        mask = srng.uniform(probs.shape) < probs
            
        return mask, probs, self.rewardFeedback(probs, mask)
    def rewardFeedback(self, probs, mask):
        mask = theano.gradient.disconnected_grad(mask)
        def f(reward, lr):
            loss = theano.gradient.disconnected_grad(reward)
            #reinf = T.mean(T.log(T.prod(probs * mask + (1-probs) * (1-mask),axis=1)) * loss)
            reinf = T.mean(T.sum(T.log(probs * mask + (1-probs) * (1-mask)),axis=1) * (loss-loss.mean()))
            #reinf = theano.printing.Print('reinf')(reinf)
            grads = T.grad(reinf, self.params)
            updates = sgd(self.params, grads, lr)
            return updates
        return f


class LazyNet:
    def __init__(self, npart, lr, reloadFrom=None, architecture=None):
        self.target = TargetNet(reloadFrom=reloadFrom,architecture=architecture)
        #self.partitionner = BanditPartitionner(npart, self.target.nhid)
        self.partitionner = ContextualBanditPartitionner(npart, self.target.nhid, self.target)

        self.comppol = ReinforceComputationPolicy(npart, self.target.nin)
        self.lr = lr
        self.npart = npart
        
    def saveTargetWeights(self, path):
        pickle.dump([i.get_value() for i in self.target.params], file(path,'w'))

        
    def trainTargetOnDataset(self, dataset, maxEpochs=50,mbsize=32,special_reg=False):
        print 'creating theano graph...'
        x = T.matrix()
        y = T.ivector()
        if not special_reg:
            o = self.target.applyToX(x, dropout=0.5)
            mbloss = T.nnet.categorical_crossentropy(o,y)
            #o = theano.printing.Print('o')(o)
            eq = T.eq(T.argmax(o,axis=1),y)
            #eq = theano.printing.Print('eq')(eq)
            acc = T.sum(eq)
            loss = T.sum(mbloss)
        else :
            o,hs = self.target.applyToX(x, dropout=0.5, return_activation=True)
            mbloss = T.nnet.categorical_crossentropy(o,y)
            #o = theano.printing.Print('o')(o)
            eq = T.eq(T.argmax(o,axis=1),y)
            #eq = theano.printing.Print('eq')(eq)
            acc = T.sum(eq)
            loss = T.sum(mbloss)


            Ws = self.target.get_weights()
            c,i,j = T.scalar(), T.scalar(dtype='int32'), T.scalar(dtype='int32')
            #h, W = T.matrix(), T.matrix()
            count_c = T.sum([T.eq(y[m],c) for m in range(mbsize)])

            #flow_ij_c = 1.0/count_c * T.sum([T.eq(y[m],c)*abs(h[m,i]*W[i,j]) for m in range(mbsize)])

            def f_ij_c(h,W,c,i,j):
                from theano import tests
                q = tests.breakpoint.PdbBreakpoint('qweqwe')
                qq = q(1,([abs(h[m,i]*W[i,j])for m in range(mbsize)]) )
                return qq
#                return 1.0/count_c * T.sum([T.eq(y[m],c)*abs(h[m,i]*W[i,j]) for m in range(mbsize)])


 #           f_ij_c = theano.function([h,W,c,i,j,y],flow_ij_c)

            nhid, nout = self.target.nhid, self.target.nout
#            fancy_reg = T.sum([T.sum([T.prod([f_ij_c(h,ws,cls,i,j,y) for cls in range(nout)]) for i,j in product(range(nhid[l]),range(nhid[l+1]))]) for l,(h,w) in enumerate(zip(hs,Ws))])

            fancy_reg = T.sum([T.sum([T.prod([f_ij_c(h,w,cls,i,j) for cls in range(nout)]) for i,j in product(range(nhid[l]),range(nhid[l+1]))]) for l,(h,w) in enumerate(zip(hs,Ws))])


#            fancy_reg = T.sum([T.prod([f_ij_c(i,j,cls) for cls in range(self.nclass)]) for i,j in all_valid_pairs_of_nodes])
            loss = loss + 0.001 * fancy_reg

        updates = sgd(self.target.params, T.grad(loss, self.target.params), self.lr)
        print 'compiling'
        learn = theano.function([x,y],[loss, acc],updates=updates)
        test = theano.function([x,y],[loss, acc])
        tolerance = 10
        last_validation_loss = 1
        for epoch in range(maxEpochs):
            train_loss, train_acc = dataset.runEpoch(dataset.trainMinibatches(mbsize), learn)
            valid_loss, valid_acc = dataset.runEpoch(dataset.validMinibatches(mbsize), test)
            print epoch, train_loss, train_acc, valid_loss, valid_acc
            if valid_loss > last_validation_loss:
                tolerance -= 1
                if tolerance <= 0:
                    break
            last_validation_loss = valid_loss
        
    def performUpdate(self, dataset, maxEpochs=50):
        print 'creating theano graph...'
        x = T.matrix()
        y = T.ivector()
        partition = self.partitionner.makePartition()
        partitionMask, probs, policyFeedbackMethod = self.comppol.applyAndGetFeedbackMethod(x)
        idxes = [partition[start:end]
                 for start,end in zip(np.cumsum(self.target.nhid) - self.target.nhid[0],
                                      np.cumsum(self.target.nhid))]
        masks = [T.zeros((x.shape[0], nhid))
                 for nhid in self.target.nhid]
        for l in range(len(self.target.nhid)):
            for i in range(self.npart):
                masks[l] = masks[l] + T.eq(idxes[l], i)[None, :] * partitionMask[:, i][:, None]
            masks[l] = T.gt(masks[l],0)
        o = self.target.applyToXWithMask(x, masks)

        mbloss = T.nnet.categorical_crossentropy(o,y)
        updates = policyFeedbackMethod(mbloss, self.lr)
        acc = T.sum(T.eq(T.argmax(o,axis=1),y))
        loss = T.sum(mbloss)
        print 'compiling'
        learn = theano.function([x,y],[loss, acc],updates=updates)
        test = theano.function([x,y],[loss, acc, T.sum(probs, axis=0)])
        testMlp = theano.function([x,y],[T.sum(T.eq(T.argmax(self.target.applyToX(x),axis=1),y))])

        print 'original model valid accuracy:',dataset.runEpoch(dataset.validMinibatches(), testMlp)
        print 'start valid accuracy:'
        valid_loss, valid_acc, probs = dataset.runEpoch(dataset.validMinibatches(), test)
        print valid_loss, valid_acc
        print 'training computation policy'
        tolerance = 5
        last_validation_loss = 1
        for epoch in range(maxEpochs):
            train_loss, train_acc = dataset.runEpoch(dataset.trainMinibatches(), learn)
            valid_loss, valid_acc, probs = dataset.runEpoch(dataset.validMinibatches(), test)
            print epoch, train_loss, train_acc, valid_loss, valid_acc
            print probs
            print test(numpy.float32(dataset.train[0][0:1]/255.), dataset.train[1][0:1])[2]
            print [np.mean(abs(i.get_value())) for i in self.comppol.params]
            if valid_loss > last_validation_loss:
                tolerance -= 1
                if tolerance <= 0:
                    break
            last_validation_loss = valid_loss

svhn = SVHN()
if 1:
    net = LazyNet(16, 0.00001,reloadFrom='./svhn_mlp/params.db')
    #net = LazyNet(8, 0.00001,reloadFrom='./svhn_mlp/retrained_params.pkl')
    net.performUpdate(svhn)
if 0:
    net = LazyNet(4, 0.001, architecture=[32*32*3,200,200,10])
    net.trainTargetOnDataset(svhn)
    net.saveTargetWeights('./svhn_mlp/retrained_params.pkl')
if 0 :
    net = LazyNet(4, 0.001, architecture=[32*32*3,200,200,10])
    net.trainTargetOnDataset(svhn,special_reg=True)
    net.saveTargetWeights('./svhn_mlp/trained_params2.pkl')