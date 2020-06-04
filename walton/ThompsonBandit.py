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
import time

run_warfarin = False
run_synth = True
# WARFARIN DATA PREP
# warfarin data import
wd = pd.read_csv('../warfarin_data.csv', header=None)

# format data
X0 = wd.to_numpy()
# correct dosage bucket (arm)
y = X0[:,-2]
# correct dosage exact amount (continuous var)
y_val = X0[:,-1]
# features
X0 = X0[:,:-2]
X = (X0 - X0.mean(axis=0))/X0.std(axis=0)
N, k = X.shape
X = np.concatenate((np.ones((N,1)),X),axis=1)
N, k = X.shape
xbar = X.mean(axis=0)
# number of arms
arms = np.unique(y).astype(int)
n_arms = arms.shape[0]

s2 = .01
rho = .1

# 0-1 reward
def reward01(y_pull, y_true):
    return (y_pull == y_true)*(1.)

# empirical rewards
yrew = np.zeros((N,n_arms))
for y_pull in arms:
    yrew[:,y_pull] = reward01(y_pull,y)

# directional prior
mu = yrew.sum(axis=0)/len(yrew)

def posterior(Sigma, mu, y, x, s2):
    Omega = np.linalg.inv(Sigma)
    Sigma_tilde = np.linalg.inv(Omega + np.outer(x, x)/s2)
    mu_tilde = Sigma_tilde @ (Omega @ mu + y/s2 * x)
    return [Sigma_tilde, mu_tilde]

def draw_post(Sigma, mu):
    return np.random.multivariate_normal(mu, Sigma)

def pull(n_arms, Sigmas, mus, yt, xt):
    betas = [draw_post(Sigmas[i], mus[i]) for i in range(n_arms)]
    fits = [(xt @ beta) for beta in betas]
    arm = np.argmax(fits)
    reward = yt[arm]
    return [arm, reward]

def step(n_arms, Sigmas, mus, yt, xt, s2, infer='full', prior=None):
    arm, reward = pull(n_arms, Sigmas, mus, yt, xt)
    Sigmas[arm], mus[arm] = posterior(Sigmas[arm], mus[arm], yt[arm], xt, s2)
    if yt[arm]==1 and infer!='none':
        for i in range(n_arms):
            if i!=arm:
                Sigmas[i], mus[i] = posterior(Sigmas[i], mus[i], 0, xt, s2)
    if yt[arm]==0 and infer=='full':
        denom = prior.sum() - prior[arm]
        for i in range(n_arms):
            if i!=arm:
                Sigmas[i], mus[i] = posterior(Sigmas[i], mus[i], prior[i]/denom, xt, s2)
    return [arm, reward]

def ThompsonBandit(y, X, Sigma, mu, s2, infer='full', prior=mu):
    N, k = X.shape
    state = np.zeros((N,2))
    for t in range(N):
        #if t % 500 == 0: print(t)
        state[t,:] = step(n_arms, Sigma, mu, y[t,:], X[t,:], s2, infer=infer, prior=prior)
    return state

def make_col_vec(x):
    N = x.shape[0]
    return x.reshape((N,1))

def init(k, n_arms, mu=None, xbar=None, p=1, rho=.5, ptype='vanilla'):
    if ptype=='vanilla':
        Sigma = p*np.eye(k)
        nu = np.zeros((k,))
        Sigmas = [Sigma for i in range(n_arms)]
        nus = [nu for i in range(n_arms)]
    elif ptype=='soft':
        S = p*np.eye(k-1)
        Sigma = np.concatenate(
                [make_col_vec(np.concatenate([np.array([(xbar @ S @ xbar) + rho]), S @ xbar])),
                np.concatenate([make_col_vec(S @ xbar).T, S],axis=0)],
                axis=1)
        Sigmas = [Sigma for i in range(n_arms)]
        nus = []
        for i in range(n_arms):
            nu = np.zeros(k)
            nu[0] = -mu[i]
            nus.append(nu)
    return [Sigmas, nus]

def cumloss(states):
    pts = np.arange(states.shape[0])
    return 1 - states[:,1]/(1+pts)

def shuffle(N, replace=False):
    idx = np.random.choice(N, size=(N,), replace=replace)
    return idx

def runorig(idx):
    S, v = init(k, n_arms, mu=mu, xbar=xbar[1:], ptype='vanilla')
    print('sim running')
    return ThompsonBandit(yrew[idx,:], X[idx,:], S, v, s2, infer='full', prior=mu)

def runsoft(idx):
    S, v = init(k, n_arms, mu=mu, xbar=xbar[1:], rho=rho, ptype='soft')
    print('sim running')
    return ThompsonBandit(yrew[idx,:], X[idx,:], S, v, s2, infer='full', prior=mu)

def runbest(idx):
    return yrew[idx,:].argmax(axis=1)

nsim = 5
draws = [shuffle(N) for i in range(nsim)]
pool = Pool(2*os.cpu_count())

if run_warfarin:
    print('running warfarin')
    t0 = time.time()
    res_soft = pool.map(runsoft, draws)
    t1 = time.time()
    print('Soft: {} seconds'.format(t1-t0))
    res_vanilla = pool.map(runorig, draws)
    t2 = time.time()
    print('Vanilla: {} seconds'.format(t2-t1))
    best_arm = pool.map(runbest, draws)
    t3 = time.time()
    print('Best: {} seconds'.format(t3-t2))

def make_frame(res, i, label='none'):
    x = pd.DataFrame(res[i])
    x.reset_index(inplace=True)
    x.columns = ['Iteration','Decision','Reward']
    x['Best'] = best_arm[i]
    x['Cumulative Regret'] = (1+x['Iteration']) - x['Reward'].cumsum()
    x['Frac Regret'] = x['Cumulative Regret']/(1+x['Iteration'])
    x['Permutation'] = i
    x['Bandit'] = label
    x = x[['Permutation','Bandit','Iteration','Decision','Best','Reward','Cumulative Regret','Frac Regret']]
    return x

def thompsonagg(t):
    tagg = t.groupby('Iteration')['Frac Regret'].agg(
        ['mean',lambda x: np.quantile(x,.025),lambda x: np.quantile(x,.975)]).reset_index()
    tagg.columns = ['Iteration','Avg','Lower','Upper']
    return tagg

def thompsonplot(t, label):
    plt.plot(t['Iteration'],t['Avg'], label=label)
    plt.fill_between(t['Iteration'],t['Lower'],t['Upper'], alpha=.3)

if run_warfarin:
    ts = pd.concat([make_frame(res_soft, i, label='Thompson Constr') for i in range(nsim)])
    tv = pd.concat([make_frame(res_vanilla, i, label='Thompson Orig') for i in range(nsim)])
    dtsagg = thompsonagg(ts)
    tvagg = thompsonagg(tv)

    thompsonplot(tsagg, 'Thompson Constr')
    thompsonplot(tvagg, 'Thompson Orig')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Frac Incorrect')
    plt.ylim(.2, .8)
    plt.grid()
    plt.savefig('thompson_warfarin.png')
    plt.close()

    tsagg.to_csv('thompson_warfarin_constr.csv', index=False)
    tvagg.to_csv('thompson_warfarin_orig.csv', index=False)

sX = pd.read_csv('synth_X.csv', header=None)
sy = pd.read_csv('synth_Y.csv', header=None)

X0 = sX.to_numpy()
# correct dosage bucket (arm)
y = sy.to_numpy().flatten()
# features
X0 = X0[:,:-2]
X = (X0 - X0.mean(axis=0))/X0.std(axis=0)
N, k = X.shape
X = np.concatenate((np.ones((N,1)),X),axis=1)
N, k = X.shape
xbar = X.mean(axis=0)
# number of arms
arms = np.unique(y).astype(int)
n_arms = arms.shape[0]
yrew = np.zeros((N,n_arms))
for y_pull in arms:
    yrew[:,y_pull] = reward01(y_pull,y)

s2 = .01
rho = .1
mu = np.array([0.1, 0.7, 0.2])

nsim = 10

def runorig(idx):
    S, v = init(k, n_arms, mu=mu, xbar=xbar[1:], p=5, ptype='vanilla')
    print('sim running')
    return ThompsonBandit(yrew[idx,:], X[idx,:], S, v, s2, infer='full', prior=mu)

def runsoft(idx):
    S, v = init(k, n_arms, mu=mu, xbar=xbar[1:], rho=rho, p=5, ptype='soft')
    print('sim running')
    return ThompsonBandit(yrew[idx,:], X[idx,:], S, v, s2, infer='full', prior=mu)

def runbest(idx):
    return yrew[idx,:].argmax(axis=1)

draws = [shuffle(N) for i in range(nsim)]

if run_synth:
    print('running synth')
    t0 = time.time()
    res_soft = pool.map(runsoft, draws)
    t1 = time.time()
    print('Soft: {} seconds'.format(t1-t0))
    res_vanilla = pool.map(runorig, draws)
    t2 = time.time()
    print('Vanilla: {} seconds'.format(t2-t1))
    best_arm = pool.map(runbest, draws)
    t3 = time.time()
    print('Best: {} seconds'.format(t3-t2))

pool.close()

if run_synth:
    ts = pd.concat([make_frame(res_soft, i, label='Thompson Constr') for i in range(nsim)])
    tv = pd.concat([make_frame(res_vanilla, i, label='Thompson Orig') for i in range(nsim)])

    tsagg = thompsonagg(ts)
    tvagg = thompsonagg(tv)

    thompsonplot(tsagg, 'Thompson Constr')
    thompsonplot(tvagg, 'Thompson Orig')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Frac Incorrect')
    plt.ylim(.2, .8)
    plt.grid()
    plt.savefig('thompson_synth.png')
    plt.close()

    tsagg.to_csv('thompson_synth_constr.csv', index=False)
    tvagg.to_csv('thompson_synth_orig.csv', index=False)

