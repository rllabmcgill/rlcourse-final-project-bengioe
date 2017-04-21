import matplotlib.pyplot as pp
import numpy
import numpy as np
import os
import cPickle as pkl

def ls(c, endswith=''):
    return [os.path.join(c, i) for i in os.listdir(c) if i.endswith(endswith)]


def getTrainExps():
    for i in ls('results', endswith='.exp'):
        exp = pkl.load(open(i))
        if exp['mode'] == 'train':
            print exp
            yield exp, i[:-4]

def getPhase2Exps(onlyWithResults=True):
    for i in ls('results', endswith='.exp'):
        exp = pkl.load(open(i))
        if exp['mode'] == 'phase2':
            if onlyWithResults:
                if os.path.exists(i[:-4]+'.results'):
                    yield exp, i[:-4]
            else:
                yield exp, i[:-4]

def getPhase2Results():
    for i in ls('results', endswith='.exp'):
        exp = pkl.load(open(i))
        if exp['mode'] == 'phase2' and os.path.exists(i[:-4]+'.results'):
            yield i[:-4], (exp, pkl.load(open(i[:-4]+'.results')))







def plot_accs_vs_nunits(exps, texp):
    acc = [r['valid_acc'] for e,r in exps.itervalues()]
    npart = [texp[e['targetnet']]['nhid']*texp[e['targetnet']]['nlayers'] for e,r in exps.itervalues()]
    pp.clf()
    pp.scatter(npart, acc)
    pp.savefig('npart_vs_nunits.png')

def plot_accs_vs_npart(exps, texp):
    acc = [r['vaccs'] for e,r in exps.itervalues()]
    npart = [e['npart'] for e,r in exps.itervalues()]
    pp.clf()
    for i,j in zip(acc,npart):
        pp.plot(i, color=('r' if j==8 else 'g'),alpha=0.5)
        
    pp.savefig('npart_vs_accs.png')

def plot_accs_vs_random_dropout(exps, texp):
    acc = [r['vaccs'] for e,r in exps.itervalues()]
    npart = [texp[e['targetnet']]['random_dropout'] for e,r in exps.itervalues()]
    pp.clf()
    for i,j in zip(acc,npart):
        pp.plot(i, color=('r' if j else 'g'),alpha=0.5)
        
    pp.savefig('npart_vs_random_dropout.png')

def plot_acc_vs_npart(exps):
    acc = [r['valid_acc'] for e,r in exps.itervalues()]
    npart = [e['npart'] for e,r in exps.itervalues()]
    pp.clf()
    pp.scatter(npart, acc)
    pp.savefig('npart_vs_acc.png')


def scatter_rA_vs_B(pexp, texp, a, b, bInPexp=True):
    path = '%s_vs_%s%s.png'%(a,'P' if bInPexp else 'T', b)
    print path
    A = [np.mean(r[a]) for e,r in pexp.itervalues()]
    if b == 'nunits':
        B = [texp[e['targetnet']]['nhid']*texp[e['targetnet']]['nlayers'] for e,r in pexp.itervalues()]
    elif bInPexp:
        B = [np.mean(e[b]) for e,r in pexp.itervalues()]
    else:
        B = [np.mean(texp[e['targetnet']][b]) for e,r in pexp.itervalues()]
    pp.clf()
    pp.scatter(B, A)
    pp.xlabel(b)
    pp.ylabel(a)
    pp.savefig(path)


def print_table(pexp, texp):
    ratios = [[[0.6/0.945,0.4/0.91,0.35/0.93],[]],[[],[]],[[],[]]]
    valid = [[[0.6,0.4,0.35],[],[0.93,0.91,0.945]],[[],[],[]],[[],[],[]]]
    orig = [[],[]]
    for e,r in pexp.itervalues():
        tnet = texp[e['targetnet']]
        print tnet
        rd = 1 if tnet['random_dropout'] else (0 if tnet.get('no_dropout') else 2)
        pga = 1 if 'DPG' in e['comppol'] else 0
        valid[rd][pga].append(r['valid_acc'])
        ratios[rd][pga].append(r['valid_acc']/r['oacc'][0])
        valid[rd][2].append(r['oacc'][0])
        #print r['valid_acc'],r['oacc']
    for name,T,mult,headers in [
            ['accuracy ratio',ratios,1,['REINFORCE','DPG']],
            ['validation accuracy (\\%)',valid,100,['REINFORCE','DPG','Original']]]:
        print map(lambda x:map(len,x), T)
        k = 5
        T = map(lambda x:map(lambda y:sorted(y)[-k:], x), T)
        print T
        mean = map(lambda x:map(np.mean, x), T)
        std = map(lambda x:map(np.std, x), T)
        fill = '%s & %s\\\\'%(name,'&'.join(headers))
        for mi,si,n in zip(mean,std,['No Dropout','0.5 Dropout','Random Dropout']):
            fill += '\hline \n' +n + '&'
            for m,s,n in zip(mi,si,headers):
                precision = 3-int(np.log(mult)/np.log(10))
                fill += '%.*f $\\pm$ %.*f &'%(precision,m*mult,precision,s*mult)
            fill = fill[:-1]+'\\\\ \n'
        s = r"""
\begin{center}
\begin{tabular}{ |%s| } 
 \hline
        %s
 \hline
\end{tabular}
\end{center}"""%('|'.join(['c']*(len(headers)+1)),fill)
        print s

if __name__ == '__main__':
    texp = dict(map(lambda x:(x[1],x[0]), list(getTrainExps())))
    pexp = dict(list(getPhase2Results()))
    
    if 0:
        for a in ['train_acc', 'train_loss', 'valid_acc', 'valid_loss', 'pmeans']:
            for b in ['npart', 'lr','nunits']:
                scatter_rA_vs_B(pexp, texp, a, b)
        plot_accs_vs_random_dropout(pexp, texp)
        plot_accs_vs_nunits(pexp, texp)
        plot_accs_vs_npart(pexp, texp)
        plot_acc_vs_npart(pexp)
        
    print_table(pexp, texp)
