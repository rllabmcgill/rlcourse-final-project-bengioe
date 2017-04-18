import numpy
import numpy as np

import theano
import theano.tensor as T
from theano.tests.breakpoint import PdbBreakpoint

from util import make_param, srng, sgd, SVHN

import shelve
import cPickle as pickle

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
                
            self.nhid = [self.params[i*2].get_value().shape[1] for i in range(len(self.params)/2-1)]
            self.nin = self.params[0].get_value().shape[0]
            
        elif architecture is not None:
            print 'making net from scratch',architecture
            for inp,out in zip(architecture[:-1],architecture[1:]):
                self.params.append(make_param((inp,out)))
                self.params.append(make_param((out,)))
                print (inp,out)
            self.nhid = architecture[:-1]
            self.nin = architecture[0]
        else:
            raise ValueError()
        
    def applyToXWithMask(self, x, masks):
        o = x
        assert len(self.params)/2-1 == len(masks)
        for i,m in enumerate(masks):
            W,b = self.params[i*2:i*2+2]
            o = T.nnet.relu(T.dot(o,W)+b) * m
            #o,m = PdbBreakpoint('test')(1, o, T.as_tensor_variable(m))
        i += 1
        W,b = self.params[i*2:i*2+2]
        o = T.nnet.softmax(T.dot(o,W)+b)
        return o

    def applyToX(self, x, dropout=None):
        o = x
        for i in range(len(self.params)/2-1):
            W,b = self.params[i*2:i*2+2]
            o = T.nnet.relu(T.dot(o,W)+b)
            if dropout is not None:
                o = o * (srng.uniform(o.shape) < dropout)# / dropout
        i += 1
        W,b = self.params[i*2:i*2+2]
        o = T.nnet.softmax(T.dot(o,W)+b)
        return o

class BanditPartitionner:
    def __init__(self, npart, nhid):
        self.n = sum(nhid)
        self.total_rewards = np.zeros((sum(nhid), npart)) + 1e-3
        self.visits = np.zeros((sum(nhid), npart)) + 1e-5
        
    def makePartition(self):
        # greedy partition
        return randargmax(self.total_rewards / self.visits, axis=1), lambda *x: []

    def partitionFeedback(self, partition, reward):
        self.total_rewards[np.ogrid[:self.n], partition] += reward
        self.visits[np.ogrid[:self.n], partition] += 1
        
class UCBBanditPartitionner:
    def __init__(self, npart, nhid, lr = 0.1, c=2):
        self.lr = lr
        self.c = c
        self.n = sum(nhid)
        self.Q = np.zeros((sum(nhid), npart)) + 1e-3
        self.visits = np.zeros((sum(nhid), npart)) + 1e-5
        self.t = 1
        
    def makePartition(self):
        # greedy partition
        return randargmax(self.Q + self.c*np.sqrt(np.log(self.t) / self.visits), axis=1), lambda *x: []

    def partitionFeedback(self, partition, reward):
        self.Q[np.ogrid[:self.n], partition] = self.Q[np.ogrid[:self.n], partition] + self.lr * (reward - self.Q[np.ogrid[:self.n], partition])
        self.visits[np.ogrid[:self.n], partition] += 1
        self.t += 1




def sample_gumbel(shape, eps=1e-5): 
  """Sample from Gumbel(0, 1)"""
  U = srng.uniform(shape)
  return -T.log(-T.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(logits.shape)
  return T.nnet.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = logits.shape[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = T.eq(y,T.max(y,1,keepdims=True))
    y = theano.gradient.disconnected_grad(y_hard - y) + y
  return y

class GumbelSoftmaxPartitionner:
    def __init__(self, npart, nhid):
        self.logits = make_param((sum(nhid), npart))
        self.npart = npart
        self.params = [self.logits]
        
    def makePartition(self):
        partition = gumbel_softmax(self.logits, 0.05, hard=True)
        #partition = T.sum(partition * T.arange(0, self.npart)[None, :], axis=1)
        #partition = PdbBreakpoint('part')(1, partition)
        def updates(loss, lr):
            return sgd([self.logits], T.grad(T.mean(loss), [self.logits]), lr)
        return partition, updates

    def partitionFeedback(self, partition, reward):
        pass

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
            #reg = T.mean((probs-0.25)**2)
            tau = 0.25
            lambda_s = 10
            reg = lambda_s * T.mean((T.mean(probs, axis=0) - tau)**2) + T.mean((T.mean(probs, axis=1) - tau)**2)
            grads = T.grad(reinf+reg, self.params)
            updates = sgd(self.params, grads, lr)
            return updates
        return f
    
class DPGComputationPolicy:
    def __init__(self, npart, nin):
        self.W = make_param((nin, npart))
        self.b = make_param((npart,))
        self.params = [self.W, self.b]
        self.V = TargetNet(architecture=[nin+npart, 200, 200, 1])
        self.params = [self.W,self.b]+self.V.params
        
    def applyAndGetFeedbackMethod(self, x):
        probs = T.nnet.sigmoid(T.dot(x,self.W)+self.b) * 0.95 + 0.025
        #probs = theano.printing.Print('probs')(probs)
        mask = srng.uniform(probs.shape) < probs
        Q = self.V.applyToX(T.concatenate([x,probs],axis=1)).flatten()
        
        def updates(loss, lr):
            qloss = T.mean(abs(Q-loss))
            tau = 0.25
            lambda_s = 100
            reg = lambda_s * T.mean((T.mean(probs, axis=0) - tau)**2) + T.mean((T.mean(probs, axis=1) - tau)**2)
            return sgd(self.params, T.grad(Q + reg, self.params), lr) +\
                sgd(self.V.params, T.grafd(qloss, self.V.params), lr)
            
        return mask, probs, updates

class LazyNet:
    def __init__(self, npart, lr, 
                 partitionner = BanditPartitionner,
                 comppol = DPGComputationPolicy,
                 reloadFrom=None, architecture=None):
        self.target = TargetNet(reloadFrom=reloadFrom,architecture=architecture)
        #self.partitionner = BanditPartitionner(npart, self.target.nhid)
        #self.partitionner = UCBBanditPartitionner(npart, self.target.nhid)
        #self.partitionner = GumbelSoftmaxPartitionner(npart, self.target.nhid)
        self.partitionner = partitionner(npart, self.target.nhid)
        #self.comppol = ReinforceComputationPolicy(npart, self.target.nin)
        #self.comppol = DPGComputationPolicy(npart, self.target.nin)
        self.comppol = comppol(npart, self.target.nin)
        self.lr = theano.shared(numpy.float32(lr))
        self.npart = npart
        
    def saveTargetWeights(self, path):
        pickle.dump([i.get_value() for i in self.target.params], file(path,'w'),-1)
    def savePartitionnerWeights(self, path):
        pickle.dump([i.get_value() for i in self.partitionner.params], file(path,'w'),-1)
    def saveComppolWeights(self, path):
        pickle.dump([i.get_value() for i in self.comppol.params], file(path,'w'),-1)

        
    def trainTargetOnDataset(self, dataset, maxEpochs=50, randomDropout=True):
        print 'creating theano graph...'
        x = T.matrix()
        y = T.ivector()
        drop = 0.5
        if randomDropout:
            drop = srng.uniform((1,),low=1e-2,high=1)
        o = self.target.applyToX(x, dropout=drop)
        mbloss = T.nnet.categorical_crossentropy(o,y)
        #o = theano.printing.Print('o')(o)
        eq = T.eq(T.argmax(o,axis=1),y)
        #eq = theano.printing.Print('eq')(eq)
        acc = T.sum(eq)
        loss = T.sum(mbloss)
        updates = sgd(self.target.params, T.grad(loss, self.target.params), self.lr)
        print 'compiling'
        learn = theano.function([x,y],[loss, acc],updates=updates)
        test = theano.function([x,y],[loss, acc])
        tolerance = 10
        last_validation_loss = 1
        vlosses = []
        vaccs = []
        for epoch in range(maxEpochs):
            train_loss, train_acc = dataset.runEpoch(dataset.trainMinibatches(), learn)
            valid_loss, valid_acc = dataset.runEpoch(dataset.validMinibatches(), test)
            vlosses.append(valid_loss); vaccs.append(valid_acc)
            print epoch, train_loss, train_acc, valid_loss, valid_acc
            if valid_loss > last_validation_loss:
                tolerance -= 1
                if tolerance <= 0:
                    break
            last_validation_loss = valid_loss
        return {'train_acc':train_acc, 'valid_acc':valid_acc,
                'train_loss':train_loss, 'valid_loss':valid_loss,
                'vlosses':vlosses, 'vaccs':vaccs,
                'last_epoch':epoch}

    def performUpdate(self, dataset, maxEpochs=100):
        #print 'creating theano graph...'
        x = T.matrix()
        y = T.ivector()
        # reset policy
        self.comppol = ReinforceComputationPolicy(self.npart, self.target.nin)
        partition, partitionFeedbackMethod = self.partitionner.makePartition()
        partitionMask, probs, policyFeedbackMethod = self.comppol.applyAndGetFeedbackMethod(x)
        if partition.ndim == 1:
            idxes = [partition[start:end]
                     for start,end in zip(np.cumsum(self.target.nhid) - self.target.nhid[0],
                                          np.cumsum(self.target.nhid))]
            masks = [T.zeros((x.shape[0], nhid))
                     for nhid in self.target.nhid]
            for l in range(len(self.target.nhid)):
                for i in range(self.npart):
                    masks[l] = masks[l] + T.eq(idxes[l], i)[None, :] * partitionMask[:, i][:, None]
                masks[l] = T.gt(masks[l],0)
        else:
            # one hot partitions
            print partition, partition.ndim
            idxes = [partition[start:end]
                     for start,end in zip(np.cumsum(self.target.nhid) - self.target.nhid[0],
                                          np.cumsum(self.target.nhid))]
            masks = [T.zeros((x.shape[0], nhid))
                     for nhid in self.target.nhid]
            for l in range(len(self.target.nhid)):
                # idxes (nhid, npart) [i,j] = 1 if i belongs in part j
                # partitionMask (mb, npart) [i,j] = 1 if part i,j should be on
                # masks (mb, nhid) [i,j] = 1 if neuron's partition is on
                masks[l] = T.sum(idxes[l].dimshuffle('x',1,0) *
                                 partitionMask.dimshuffle(0,1,'x'), axis=1)
    
        o = self.target.applyToXWithMask(x, masks)

        mbloss = T.nnet.categorical_crossentropy(o,y)
        updates = policyFeedbackMethod(mbloss, self.lr)
        updates += partitionFeedbackMethod(mbloss, self.lr)
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
        print probs
        print 'training computation policy'
        tolerance = 50
        last_validation_loss = 100
        vlosses = []
        vaccs = []
        pmeans = []
        for epoch in range(maxEpochs):
            train_loss, train_acc = dataset.runEpoch(dataset.trainMinibatches(), learn)
            valid_loss, valid_acc, probs = dataset.runEpoch(dataset.validMinibatches(), test)
            vlosses.append(valid_loss); vaccs.append(valid_acc); pmeans.append(probs.mean())
            print epoch, train_loss, train_acc, valid_loss, valid_acc
            print probs
            print test(numpy.float32(dataset.train[0][0:1]/255.), dataset.train[1][0:1])[2]
            if valid_loss > last_validation_loss:
                tolerance -= 1
                self.lr.set_value(self.lr.get_value() * numpy.float32(0.75))
                print 'new tolerance',tolerance, self.lr.get_value()
                if tolerance <= 0:
                    break
            if partition.ndim >= 2:
                print partition.eval({}).argmax(axis=1)
            last_validation_loss = valid_loss
            print self.partitionner.logits.get_value()
        self.partitionner.partitionFeedback(partition, valid_acc)
        print list(probs), probs.mean()
        return {'train_acc':train_acc, 'valid_acc':valid_acc,
                'train_loss':train_loss, 'valid_loss':valid_loss,
                'vlosses':vlosses, 'vaccs':vaccs, 
                'pmeans':pmeans,
                'last_epoch':epoch}
    
    def updateLoop(self, dataset):
        accs = []
        for i in range(100):
            accs.append(self.performUpdate(dataset))
            print '    ', i, max(accs)
            print accs

def ls(c, endswith=''):
    return [os.path.join(c, i) for i in os.listdir(c) if i.endswith(endswith)]

def run_exp(name):
    if os.path.exists(name+'.result'): print name,'has results'; return
    if os.path.exists(name+'.lock'): print name,'is being run'; return
    file(name+'.lock','w').write(str(os.getpid()))
    try:
        exp_params = pkl.load(file(name+'.exp','r'))
        print 'running', name, exp_params

        if exp_params['mode'] == 'train':
            # train target network
            net = LazyNet(2, 0.001, 
                          architecture=[32*32*3]+[exp_params['nhid']]*exp_params['nlayers']+[10])
            results = net.trainTargetOnDataset(svhn, randomDropout=exp_params['random_dropout'])
            net.saveTargetWeights(name+'.weights')
            pkl.dump(results, file(name+'.result','w'),-1)

        elif exp_params['mode'] == 'comppol':
            net = LazyNet(exp_params['npart'], 
                          exp_params['lr'],
                          partitionner=eval(exp_params['partitionner']),
                          comppol=eval(exp_params['comppol']),
                          reloadFrom=exp_params['weights'])
            results = net.performUpdate(svhn)
            net.savePartitionnerWeights(name+'.pweights')
            net.saveComppolWeights(name+'.cweights')
            pkl.dump(results, file(name+'.results','w'),-1)
    finally:
        os.remove(name+'.lock')

def generate_exps(exps):
    import uuid
    for i in exps:
        name = str(uuid.uuid4())[:8]
        path = 'results/'+name+'.exp'
        while os.path.exists(path):
            name = str(uuid.uuid4())[:8]
            path = 'results/'+name+'.exp'
            
        pkl.dump(i, file(path,'w'), -1)
    
        
svhn = SVHN()
if 0:
    #net = LazyNet(16, 0.00001,reloadFrom='./svhn_mlp/params.db')
    net = LazyNet(8, 0.0001,reloadFrom='./svhn_mlp/retrained_params.pkl')
    net.updateLoop(svhn)
if 0:
    net = LazyNet(16, 0.05,reloadFrom='./svhn_mlp/retrained_params_4_200_rd_nodiv.pkl')
    #net = LazyNet(16, 0.005,reloadFrom='./svhn_mlp/retrained_params_4_200.pkl')
    #net = LazyNet(8, 0.05,reloadFrom='./svhn_mlp/retrained_params.pkl')
    #net = LazyNet(8, 0.05,reloadFrom='./svhn_mlp/params.db')
    net.performUpdate(svhn)
if 0:
    net = LazyNet(4, 0.001, architecture=[32*32*3,200,200,200,200,10])
    net.trainTargetOnDataset(svhn)
    net.saveTargetWeights('./svhn_mlp/retrained_params_4_200_rd_l1.pkl')

import multiprocessing
import os
import os.path
import cPickle as pkl
import time

def getTrainExps():
    for i in ls('results', endswith='.exp'):
        exp = pkl.load(open(i))
        if exp['mode'] == 'train':
            yield exp, i[:-4]

if 1:
    exps = []
    for nlayers in [2,3,4]:
        for nhid in [200,400,800]:
            for random_dropout in [True,False]:
                exps.append({'mode':'train', 'nlayers':nlayers, 'nhid':nhid,
                             'random_dropout':random_dropout})
    
    if 0:
        generate_exps(exps)
    else:
        phase2 = []
        for npart in [8,16]:
            for lr in [0.05,0.005,0.001]:
                for partitionner in ['BanditPartitionner', 'GumbelSoftmaxPartitionner']:
                    for comppol in ['ReinforceComputationPolicy', 'DPGComputationPolicy']:
                        for exp, path in getTrainExps():
                            phase2.append({'mode':'phase2',
                                           'npart':npart,
                                           'lr':lr,
                                           'partitionner':partitionner,
                                           'comppol':comppol,
                                           'targetnet':path,
                                           'weights':path+'.weights'})
                            #print path
        print len(phase2)

if 0:
    pool = multiprocessing.Pool(8)
    exps = [i[:-4] for i in ls('results')
            if i.endswith('.exp')]
    pool.map(run_exp, exps)
