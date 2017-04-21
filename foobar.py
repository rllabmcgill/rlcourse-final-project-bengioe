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

                self.params = map(theano.shared, [params[i] for i in
                                                  ['Wt0_0', 'bt0_1', 'Wt1_2', 'bt1_3', 'Woutput_4', 'boutput_5']])
            else:
                self.params = map(theano.shared, pickle.load(open(reloadFrom)))
            print [i.get_value().shape for i in self.params]
                
            self.nhid = [self.params[i*2].get_value().shape[1] for i in range(len(self.params)/2-1)]
            self.nin = self.params[0].get_value().shape[0]
            self.nout = self.params[-1].get_value().shape[0]


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
        assert len(self.params)/2-1 == len(masks)
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
                o = o * (srng.uniform(o.shape) < dropout)# / dropout
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
        #self.partitionner = ContextualBanditPartitionner(npart, self.target.nhid, self.target)
        #self.comppol = ReinforceComputationPolicy(npart, self.target.nin)
        self.npart = npart
        
    def saveTargetWeights(self, path):
        pickle.dump([i.get_value() for i in self.target.params], file(path,'w'),-1)
    def savePartitionnerWeights(self, path):
        pickle.dump([i.get_value() for i in self.partitionner.params], file(path,'w'),-1)
    def saveComppolWeights(self, path):
        pickle.dump([i.get_value() for i in self.comppol.params], file(path,'w'),-1)


    def trainTargetOnDataset(self, dataset, maxEpochs=50,mbsize=32,special_reg=False,randomDropout=False,
                             doDropout=True):
        print 'creating theano graph...'
        x = T.matrix()
        y = T.ivector()
        if not special_reg:
            if doDropout: 
                if randomDropout:
                    drop = srng.uniform((1,),low=1e-2,high=1)
                else:
                    drop = 0.5
            else:
                drop = None
            o = self.target.applyToX(x, dropout=drop)
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
                return 1.0/count_c * T.sum([T.eq(y[m],c)*abs(h[m,i]*W[i,j]) for m in range(mbsize)])


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
        vlosses = []
        vaccs = []
        for epoch in range(maxEpochs):
            train_loss, train_acc = dataset.runEpoch(dataset.trainMinibatches(mbsize), learn)
            valid_loss, valid_acc = dataset.runEpoch(dataset.validMinibatches(mbsize), test)
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
        
        oacc = dataset.runEpoch(dataset.validMinibatches(), testMlp)
        print 'original model valid accuracy:',oacc
        #print 'start valid accuracy:'
        #valid_loss, valid_acc, probs = dataset.runEpoch(dataset.validMinibatches(), test)
        #print valid_loss, valid_acc
        #print probs
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
            #print self.partitionner.logits.get_value()
        self.partitionner.partitionFeedback(partition, valid_acc)
        print list(probs), probs.mean()
        return {'train_acc':train_acc, 'valid_acc':valid_acc,
                'train_loss':train_loss, 'valid_loss':valid_loss,
                'vlosses':vlosses, 'vaccs':vaccs, 
                'oacc': oacc,
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
    exp_params = pkl.load(file(name+'.exp','r'))
    #if exp_params['mode'] == 'phase2': return
    if os.path.exists(name+'.result'): print name,'has results'; return
    if os.path.exists(name+'.lock'): print name,'is being run'; print exp_params; return
    file(name+'.lock','w').write(str(os.getpid()))
    try:
        print 'running', name, exp_params

        if exp_params['mode'] == 'train':
            # train target network
            net = LazyNet(2, 0.001, 
                          architecture=[32*32*3]+[exp_params['nhid']]*exp_params['nlayers']+[10])
            results = net.trainTargetOnDataset(svhn, randomDropout=exp_params['random_dropout'], 
                                               doDropout=not exp_params.get('no_dropout',False))
            net.saveTargetWeights(name+'.weights')
            pkl.dump(results, file(name+'.result','w'),-1)

        elif exp_params['mode'] == 'phase2':
            if os.path.exists(exp_params['weights']):
                net = LazyNet(exp_params['npart'], 
                          exp_params['lr'],
                          partitionner=eval(exp_params['partitionner']),
                          comppol=eval(exp_params['comppol']),
                          reloadFrom=exp_params['weights'])
                results = net.performUpdate(svhn)
                pkl.dump(results, file(name+'.results','w'),-1)
                net.savePartitionnerWeights(name+'.pweights')
                net.saveComppolWeights(name+'.cweights')
            else:
                print 'net', exp_params['weights'], 'not trained yet'
    except Exception,e:
        import traceback
        traceback.print_exc()
        raise e
    except KeyboardInterrupt,e:
        import traceback
        traceback.print_exc()
        raise ValueError('keyboard interrupt')
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
        print name
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
    #net.saveTargetWeights('./svhn_mlp/retrained_params.pkl')

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

def getPhase2Exps():
    for i in ls('results', endswith='.exp'):
        exp = pkl.load(open(i))
        if exp['mode'] == 'phase2':
            if 'Bandit' in exp['partitionner']:
                print i
if 0:
    exps = []
    for nlayers in [2,3,4]:
        for nhid in [200,400,800]:
            exps.append({'mode':'train', 'nlayers':nlayers, 'nhid':nhid,
                         'no_dropout':True,
                         'random_dropout':None})
    exps_ = []
    for nlayers in [2,3,4]:
        for nhid in [200,400,800]:
            for random_dropout in [True,False]:
                exps.append({'mode':'train', 'nlayers':nlayers, 'nhid':nhid,
                             'random_dropout':random_dropout})
    
    if 0:
        generate_exps(exps)
    elif 0:
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
        generate_exps(phase2)
    elif 0:
        phase2 = []
        for npart in [8,16]:
            for lr in [0.05,0.005,0.001]:
                for partitionner in ['GumbelSoftmaxPartitionner']:
                    for comppol in ['ReinforceComputationPolicy', 'DPGComputationPolicy']:
                        for exp, path in getTrainExps():
                            if 'no_dropout' not in exp:
                                continue
                            print exp, path
                            phase2.append({'mode':'phase2',
                                           'npart':npart,
                                           'lr':lr,
                                           'partitionner':partitionner,
                                           'comppol':comppol,
                                           'targetnet':path,
                                           'weights':path+'.weights'})
                            #print path
        print len(phase2)
        generate_exps(phase2)

if 1:
    pool = multiprocessing.Pool(4)
    exps = [i[:-4] for i in ls('results')
            if i.endswith('.exp')]
    pool.map(run_exp, exps)

if 0:
    net = LazyNet(4, 0.001, architecture=[32*32*3,200,200,10])
    net.trainTargetOnDataset(svhn,special_reg=True)
    net.saveTargetWeights('./svhn_mlp/trained_params2.pkl')
