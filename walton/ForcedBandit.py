import pdb
import itertools
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from multiprocessing import Pool
import os

# WARFARIN DATA PREP
# warfarin data import
wd = pd.read_csv('../warfarin_data.csv', header=None)

# format data
X = wd.to_numpy()
# correct dosage bucket (arm)
y = X[:,-2]
# correct dosage exact amount (continuous var)
y_val = X[:,-1]
# features
X = X[:,:-2]
N = X.shape[0]
# normalize
X = (X - X.mean(axis=0))/X.std(axis=0)
X = np.concatenate((np.ones((N,1)),X),axis=1)
xbar = X.mean(axis=0)
k = len(xbar)
# number of arms
arms = np.unique(y).astype(int)
n_arms = arms.shape[0]

# 0-1 Loss
def loss01(y_pull, y_true):
    return (y_pull == y_true)*(-1.) + 1

# empirical losses
yloss = np.zeros((N,n_arms))
for y_pull in arms:
    yloss[:,y_pull] = loss01(y_pull,y)

# directional prior
mu = 1 - (-1*yloss + 1).sum(axis=0)/len(yloss)

def make_col_vec(x):
    N = x.shape[0]
    return x.reshape((N,1))

# Estimators
def fastOLS(y, X):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def OLSest(X, y, arm):
    beta = fastOLS(y, X)
    f = lambda x: x @ beta
    return f

def fastOLSconstr(y, X, xbar, mu):
    Xi = np.linalg.pinv(X.T @ X)
    Xy = X.T @ y
    Gamma = xbar @ Xi
    gamma = Gamma @ xbar.T
    return Xi @ Xy + Gamma.T * (Gamma @ Xy - mu)/gamma

def cOLSest(X, y, arm):
    beta = fastOLSconstr(y, X, xbar, mu[arm])
    f = lambda x: x @ beta
    return f

oracleOLSbeta = []
for i in range(n_arms):
    oracleOLSbeta.append(fastOLS(yloss[:,i], X))
oracleOLSpred = [lambda x: x @ oracleOLSbeta[0],
            lambda x: x @ oracleOLSbeta[1],
            lambda x: x @ oracleOLSbeta[2]]

def oracOLSest(X, y, arm):
    return oracleOLSpred[arm]

# FORCED SAMPLING BANDIT FUNCTIONS
# Things to keep track of:
# sampling: which arm to sample (or greedy) each period
# arm_hist: which arm has been sampled for each period
# FS_pred: current forced sample predictor
# GR_pred: current greedy sample predictor
# yhat: constructed counterfactual y's

def gen_FSS(N, k, n_arms, schedule=True, B=None, q=1, epsilon=.05, overlap=True, seed=None):
    """
    Generate Forced Sampling Procedure:
    N x k is dimensions of feature matrix X
    n_arms is number of bandit arms
    schedule: bool -- schedule or epsilon-greedy sampling
    B: int -- length of block forced sample
    q: int -- schedule sampling parameter
    epsilon: float -- greedy forced sample rate
    overlap: bool -- whether or not forced schedule overlaps with block (subsumed by block)
    seed: int -- random seed to use
    """
    if B is None:
        B = n_arms*k
    if seed is not None:
        np.random.set_seed(seed)
    block = np.tile(np.arange(n_arms),
            round(B/n_arms+.5))[:B]
    S = N - B
    if not schedule:
        greedy = np.random.binomial(1, epsilon, S)
        nforced = greedy.sum()
        greedy = greedy - 1
        greedy[greedy == 0] = np.random.randint(n_arms, size=nforced)
    else:
        greedy = (-1)*np.ones((S,))
        for i in range(1,n_arms+1):
            idx_i = np.array([(2**n-1)*n_arms*q+j for (n, j) in
                itertools.product(range(int(np.log2(N/(n_arms*1) + 1)+1)), range(q*(i-1), q*i))])
            if overlap:
                idx_i = idx_i - (B)
            idx_i = idx_i[(idx_i >= 0) & (idx_i < S)]
            greedy[idx_i] = i-1
    sampling = np.concatenate([block, greedy]).astype(int)
    arm_hist = sampling.copy()
    return [sampling, arm_hist]

def update_predictor(t, arm, sampling, arm_hist, X, y, estimator, greedy=False, infer='full'):
    if greedy:
        track = arm_hist
    else:
        track = sampling
    if infer=='full':
        mask = (track[:t] >= 0)
    elif infer=='partial':
        mask = (track[:t] >= 0) & ((y[np.arange(t),arm_hist[:t]]==0) | (arm_hist[:t]==arm))
    else:
        mask = (track[:t] == arm)
    ysamp = y[:t][mask,arm]
    Xsamp = X[:t,:][mask,:]
    #pdb.set_trace()
    return estimator(Xsamp, ysamp, arm)

def infer_y(arm, yt, n_arms, mu, infer='full'):
    "infer: ('full','partial','none')"
    ythat = np.zeros((n_arms,))
    outcome = yt[arm]
    ythat[arm] = outcome
    if infer=='none':
        return ythat
    if outcome==0:
        ythat[np.arange(n_arms)!=arm] = 1
    elif infer=='full':
        musum = (1 - mu[np.arange(n_arms)!=arm]).sum()
        for i in range(n_arms):
            if i!=arm:
                ythat[i] = 1 - (1-mu[i])/musum
    return ythat

def pull(n_arms, samplingt, FS_pred, GR_pred, Xt, efilter=True, h=2):
    if samplingt!=-1:
        return samplingt
    else:
        fitted = np.zeros((n_arms,))
        for i in range(n_arms):
            fitted[i] = FS_pred[i](Xt)
        if efilter:
            keep_arms = np.arange(n_arms)[fitted <= (fitted.min() + h/2)]
            GR_val_min = np.Inf
            arm_current = 0
            for i in range(len(keep_arms)):
                GR_val = GR_pred[keep_arms[i]](Xt)
                if GR_val <= GR_val_min:
                    arm_current = keep_arms[i]
                    GR_val_min = GR_val
        else:
            arm_current = fitted.argmin()
        #pdb.set_trace()
        return arm_current

def step(t, B):
    st = B['sampling'][t]
    if st==-1:
        for arm in B['pending']:
            B['FS_pred'][arm] = update_predictor(t, arm, B['sampling'], B['arm_hist'], B['X'],
                    B['yhat'], B['estimator'], greedy=False, infer=B['infer'])
            B['GR_pred'][arm] = update_predictor(t, arm, B['sampling'], B['arm_hist'], B['X'],
                    B['yhat'], B['estimator'], greedy=True, infer=B['infer'])
        B['pending'] = set()
    arm = pull(n_arms, B['sampling'][t], B['FS_pred'], B['GR_pred'],
            B['X'][t,:], efilter=B['efilter'], h=B['h'])
    if st!=-1:
        B['pending'].add(arm)
    B['arm_hist'][t] = arm
    #pdb.set_trace()
    B['yhat'][t,:] = infer_y(arm, B['y'][t,:], B['n_arms'], B['mu'])
    if B['verbose']:
        print("Iter: {}, Arm: {}".format(t,arm))
        print("Greed: {}, Loss: {}".format(st==-1,B['y'][t,arm]))

def initialize(X, y, estimator, mu=None, schedule=True, B=None, q=1, h=2, epsilon=.05, overlap=True, seed=None, infer='full', efilter=True, verbose=False):
    [N, k] = X.shape
    n_arms = y.shape[1]
    if B is None:
        B = n_arms*k
    [sampling, arm_hist] = gen_FSS(N, k, n_arms,schedule=schedule,
            B=B, q=q, epsilon=epsilon, overlap=overlap, seed=seed)
    FS_pred = [0 for i in range(n_arms)]
    GR_pred = [0 for i in range(n_arms)]
    pending = set(range(n_arms))
    bandit = {'sampling': sampling, 'arm_hist': arm_hist, 'n_arms': n_arms, 'FS_pred': FS_pred, 'GR_pred': GR_pred,
            'X': X, 'y': y, 'yhat': y.copy(), 'pending': pending, 'estimator': estimator,
            'mu': mu, 'infer': infer, 'efilter': efilter, 'h': h, 'verbose': verbose}
    return bandit

def run_bandit(B):
    [N, k] = B['X'].shape
    for t in range(N):
        step(t, B)
    return B['arm_hist']

def emploss(B):
    N = B['y'].shape[0]
    return B['y'][np.arange(N), B['arm_hist']]

def plotloss(B):
    el = emploss(B)
    N = len(el)
    plt.plot(np.arange(N), el.cumsum()/(1+np.arange(N)))

def simlosses(sims, yloss):
    N = yloss.shape[0]
    pts = np.arange(N)
    sl = [make_col_vec(yloss[pts,s]) for s in sims]
    return np.concatenate(sl, axis=1)

def plotsimloss(s, ax=None, show_conf_bounds=True, label='OLSbandit'):
    sims = simlosses(s, yloss)
    if ax is None:
        fig, ax = plt.subplots()
    cs = sims.cumsum(axis=0)
    cs_avg = cs.mean(axis=1)
    cs_low = np.quantile(cs, .025, axis=1)
    cs_high = np.quantile(cs, .975, axis=1)
    points = np.arange(sims.shape[0])
    ax.plot(points, cs_avg/(points+1), label=label)
    if show_conf_bounds:
        ax.fill_between(points,
                    cs_low/(points+1),
                    cs_high/(points+1),
                alpha=0.2)
    return ax

def shuffle(N, replace=False):
    idx = np.random.choice(N, size=(N,), replace=replace)
    return idx

def bOLSegreed(idx):
    bandit = initialize(X[idx,:], yloss[idx,:], OLSest, mu=mu, infer='none',
            schedule=False, epsilon=.1, h=1, efilter=False)
    return run_bandit(bandit)

def bOLSsched(idx):
    banditsch = initialize(X[idx,:], yloss[idx,:], OLSest, mu=mu, infer='none',
            schedule=True, h=5, efilter=True)
    return run_bandit(banditsch)

def bOlsSchPart(idx):
    bandit_inf = initialize(X[idx,:], yloss[idx,:], OLSest, mu=mu, infer='partial',
            schedule=True, h=3, efilter=True)
    return run_bandit(bandit_inf)

def bOlsSchFullShort(idx):
    bandit_sh = initialize(X[idx,:], yloss[idx,:], OLSest, mu=mu, infer='full',
            B=k, schedule=True, h=5, efilter=True)
    run_bandit(bandit_sh)

def bOlsOracle(idx):
    oracband = initialize(X[idx,:], yloss[idx,:], oracOLSest, mu=mu, B=0, schedule=False, epsilon=0, efilter=False)
    return run_bandit(oracband)

n_sim = 10
draws = [shuffle(N) for i in range(n_sim)]

pool = Pool(2*os.cpu_count())

oeg = pool.map(bOLSegreed, draws)
osch = pool.map(bOLSsched, draws)
oscp = pool.map(bOlsSchPart, draws)
oscfs = pool.map(bOlsSchFullShort, draws)
oorac = pool.map(bOlsOracle, draws)

pool.close()

a1 = plotsimloss(oorac, label='OLS oracle')
plotsimloss(oeg, ax=a1, label='OLS .1-greedy')
plotsimloss(osch, ax=a1, label='OLS bandit')
plotsimloss(oscp, ax=a1, label='OLS infer')
plotsimloss(oscfs, ax=a1, label='OLS full infer short')
a1.set_xlabel('Observations')
a1.set_ylabel('Fraction Incorrect')
plt.legend()
plt.ylim(.2,.8)
plt.grid()
plt.show()
