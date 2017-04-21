import time
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pp

import theano
import theano.tensor as T
import numpy as np
import pickle
from foobar import TargetNet as NNet, svhn
from util.plot_results import ls, getTrainExps, getPhase2Exps, getPhase2Results

if False :
    texp = dict(map(lambda x:(x[1],x[0]), list(getTrainExps())))
    pexp = dict(list(getPhase2Results()))

    pr = getPhase2Results()

    argmax = ''
    maxvac =0
    for name, (exp, results) in pr:
        t = texp[exp['targetnet']]
        if results['oacc'][0] > maxvac and t['random_dropout']:
            argmax = exp, t
            maxvac = results['oacc'][0]
            uname = name
    print maxvac, argmax, uname
    exit()

    nnet = NNet(reloadFrom=argmax[0]['weights'])


if True :
    nnet = NNet(reloadFrom="small_mlp_3_100.pkl")

    # coactivation probabilities
    x = T.matrix()
    o,hs = nnet.applyToX(x, return_activation=True)

    nhid = nnet.nhid
    N = sum(nnet.nhid)
    coact = np.zeros((N,N))
    print coact.shape, N*N, nhid

    extract = theano.function([x], hs[1:])

if 0:
    mb = svhn.trainMinibatches()
    n = mb.next()
    t0 = time.time()
    _ = 0
    print n, n/32
    for x,y in mb:
        hs = extract(x)
        for h, nh, idx in zip(hs,nhid, np.cumsum(nhid)):
            for i in range(32):
                gt0 = h[i] > 0
                coact[idx-nh:idx, idx-nh:idx] += np.outer(gt0,gt0)
        _ += 1
        print '\b',_, time.time()-t0, '\r\b',
        sys.stdout.flush()
        if _ > 50: break
    print 
    pp.matshow(coact)
    pp.savefig('coact_blocks.png')
    pp.matshow(np.log(coact+1))
    pp.savefig('coact_blocks_log.png')
    print time.time()-t0


if 0:
    coact_full = np.zeros((N,N))

    mb = svhn.trainMinibatches()
    n = mb.next()
    t0 = time.time()
    _ = 0
    for x,y in mb:
        hs = extract(x)
        for i in range(32):
            H = np.concatenate([h[i] for h in hs])
            gt0 = H > 0
            coact_full += np.outer(gt0,gt0)
        _ += 1
        print '\b',_, time.time()-t0, '\r\b',
        sys.stdout.flush()
        if _ > 50: break
    print 
    pp.matshow(coact_full)
    pp.savefig('coact_full.png')
    pp.matshow(np.log(coact_full+1))
    pp.savefig('coact_full_log.png')
    print time.time()-t0



if 1:
    coact_full = np.zeros((N,N))

    mb = svhn.trainMinibatches()
    n = mb.next()
    t0 = time.time()
    _ = 0
    for x,y in mb:
        hs = extract(x)
        for i in range(32):
            H = np.concatenate([h[i] for h in hs])
            coact_full += np.outer(H,H)
        _ += 1
        print '\b',_, time.time()-t0, '\r\b',
        sys.stdout.flush()
        if _ > 100: break
    #coact_full[range(N), range(N)] = 0
    means = coact_full.mean(axis=1)
    coact_full = coact_full[sorted(range(N),key=lambda x:means[x])]
    means = coact_full.mean(axis=0)
    coact_full = coact_full[:,sorted(range(N),key=lambda x:means[x])]
    pp.matshow(coact_full)
    pp.savefig('coprod_full.png')
    pp.matshow(np.log(coact_full+1))
    pp.savefig('coprod_full_log.png')
    print time.time()-t0

if 0:
#    f = open("results_flow3.pkl","rb")
    f = open("results_flow_2_130.pkl","rb")
    loaded_flow = pickle.load(f)
    print map(lambda x:x.shape, loaded_flow)
#    assert False

    hs = []
    for c in range(10):
        print('c',c)

        H = np.concatenate([lf[c].flatten() for lf in loaded_flow])
#        H = np.concatenate([np.sum(lf[c],axis=0) for lf in loaded_flow])
        hs.append(H)

    for f in loaded_flow:
        f = f.prod(axis=0)
        pp.matshow(f)

    #    if _ > 2000: break
    #coact_full[range(N), range(N)] = 0
    for c in range(10):
        H = hs[c]
        coact_full = np.outer(H,H)
#        coact_full = matrixes[c]
        means = coact_full.mean(axis=1)
#        coact_full = coact_full[sorted(range(coact_full.shape[0]),key=lambda x:means[x])]
        pp.matshow(coact_full)
        pp.savefig('coprod_full_%i.png'%c)
        pp.matshow(np.log(coact_full+1))
        pp.savefig('coprod_full_log_%i.png'%c)
    print time.time()-t0

if 0 :
#    f = open("results_flow_2_130.pkl","rb")
    f = open("results_flow3.pkl","rb")
    loaded_flow = pickle.load(f)


    for i,f in enumerate(loaded_flow):
        f = np.sum(np.log(f+f[f>0].min()),axis=0)
        f = f[sorted(range(f.shape[0]),key=lambda x : f[x].mean())]
        pp.matshow(f)
        pp.savefig('coprod_full_%i.png'%i)

