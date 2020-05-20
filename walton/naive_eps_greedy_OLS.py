#!/usr/bin/env python
# coding: utf-8

import pdb
import itertools
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

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
X = np.concatenate((np.ones((N,1)),X),axis=1)
# number of arms
arms = np.unique(y).astype(int)
n_arms = arms.shape[0]

def make_col_vec(x):
    N = x.shape[0]
    return x.reshape((N,1))

# 0-1 Loss
def loss01(y_pull, y_true):
    return (y_pull == y_true)*(-1.) + 1

def fastOLS(y, X):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

# include lasso and ridge and maybe random forest

# choose arm based on linear combination x'beta
def linear_arm(params, x):
    losses = []
    for B in params:
        losses.append(np.asmatrix(x @ B).T)
    losses = np.concatenate(losses, axis=1)
    return np.asarray(losses.argmin(axis=1)).flatten()

# empirical losses
yloss = np.zeros((N,n_arms))
for y_pull in arms:
    yloss[:,y_pull] = loss01(y_pull,y)

class FS_Bandit:
    """Forced Sampling Bandit: includes an initial block of forced sampling followed by
     epsilon-greedy sampling 'sample' or the forced sampling schedule 'schedule' of [REF]. User provides
     the estimator (default is OLS) and selector (default is linear)"""

    def __init__(self, yloss, X, sample='epsilon', gr_filter=False, estimator=fastOLS,
            selector=linear_arm, forced_samples=None, q=1, epsilon=.1, h=.5,
            seed=None, oracle=True):
        self.yloss0 = yloss
        self.X0 = X
        self.sample = sample
        self.gr_filter = gr_filter
        self.estimator = estimator
        self.selector = selector
        self.N = X.shape[0]
        self.k = X.shape[1]
        self.n_arms = yloss.shape[1]
        self.q = q
        self.epsilon = epsilon
        self.h = h
        # default number of initial forced samples is 1/4 of the data
        # divided equally among the arms
        if forced_samples is None:
            forced_samples = int(N * .25/ n_arms)
        self.forced_samples = forced_samples
        self.reset_data()
        self.set_seed(seed)
        if oracle:
            self.run_oracle = True
            self.oracle()

    def set_seed(self, seed=None):
        self.seed = seed
        if self.seed is not None:
            np.random.seed(seed)

    def set_sample_procedure(self):
        block = np.tile(np.arange(self.n_arms),
                        self.forced_samples)
        if self.sample=='epsilon':
            greedy = np.random.binomial(1, self.epsilon, self.N - self.forced_samples*self.n_arms)
            nforced = greedy.sum()
            greedy = greedy - 1
            greedy[greedy == 0] = np.random.randint(self.n_arms, size=nforced)
        elif self.sample=='schedule':
            greedy = self.schedule()
        else:
            raise TypeError('Sampling needs to be random epsilon or schedule')
        self.sampling = np.concatenate([block, greedy]).astype(int)
        self.arm_hist = self.sampling.copy()

    def schedule(self):
        N = self.N
        n_arms = self.n_arms
        fN = self.forced_samples*n_arms
        gN = N - fN
        greedy = (-1)*np.ones((gN,))
        q = self.q
        for i in range(1,n_arms+1):
            idx_i = np.array([(2**n-1)*n_arms*q+j for (n, j) in
                itertools.product(range(int(np.log2(N/(n_arms*1) + 1)+1)), range(q*(i-1), q*i))])
            idx_i = idx_i - (fN)
            idx_i = idx_i[(idx_i >= 0) & (idx_i < gN)]
            greedy[idx_i] = i-1
        return greedy

    def reset_data(self):
        # reset data back to original
        self.X = self.X0.copy()
        self.yloss = self.yloss0.copy()

    def reset(self):
        # preallocate space for all forced sample arrays
        self.X_arm_FSS = []
        self.yloss_arm_FSS = []
        self.X_arm_GR = []
        self.yloss_arm_GR = []
        self.param = []
        self.param_needs_update = []
        for i in range(self.n_arms):
            # features and losses corresponding to each arm
            self.X_arm_FSS.append(self.X[self.sampling==i,:])
            self.yloss_arm_FSS.append(self.yloss[self.sampling==i,i])
            # greedy sample preallocate all space
            self.X_arm_GR.append(np.zeros((self.N - self.forced_samples*self.n_arms, self.k)))
            self.yloss_arm_GR.append(np.zeros(self.N - self.forced_samples*self.n_arms))
            self.param.append(np.zeros((self.k,)))
            self.param_needs_update.append(True)
        # current index of sample
        self.t = 0

    def resample(self, replace=True):
        # resampling of data for bootstrap or permutation
        idx = np.random.choice(self.N, size=(self.N,), replace=replace)
        self.X = self.X0[idx,:]
        for j in range(self.n_arms):
            self.yloss[:,j] = self.yloss0[idx,j]

    def oracle(self):
        # oracle parameters
        oracle_params = []
        for i in range(n_arms):
            mod = self.estimator(self.yloss[:,i], self.X)
            oracle_params.append(mod)
        self.oracle_params = oracle_params
        # oracle arm choices
        self.oracle_arm = self.selector(self.oracle_params, self.X).flatten()
        # oracle loss
        self.oracle_loss = self.yloss[np.arange(self.N),self.oracle_arm]

    def next_action(self):
        arm = self.sampling[self.t]
        # greedy is denoted by arm = -1
        if arm==-1:
            self.do_greedy()
        else:
            self.do_forced()
        self.t += 1

    def do_greedy(self):
        # take the greedy action
        n_arms = self.n_arms
        self.update_param()
        t = self.t
        if self.gr_filter:
            fitted = np.zeros((n_arms,))
            for i in range(n_arms):
                fitted[i] = self.X[t,:] @ self.param[i]
            keep_arms = np.arange(n_arms)[fitted <= (fitted.min() + self.h/2)]
            GR_val_min = np.Inf
            arm_current = 0
            for i in range(len(keep_arms)):
                GR_val = self.X[t,:] @ self.update_greedy_param(keep_arms[i])
                if GR_val <= GR_val_min:
                    arm_current = keep_arms[i]
                    GR_val_min = GR_val
        else:
            arm_current = self.selector(self.param, self.X[t,:])[0]
        self.arm_hist[t] = arm_current
        greedy_ind = ((self.arm_hist[:(t+1)]==arm_current).sum() -
                (self.sampling[:(t+1)]==arm_current).sum())
        self.X_arm_GR[arm_current][greedy_ind,:] = self.X[t,:].reshape((1,self.k))
        self.yloss_arm_GR[arm_current][greedy_ind] = self.yloss[t,arm_current]

    def update_greedy_param(self, i):
        t = self.t
        n_to_t = (self.sampling[:(t+1)]==i).sum()
        g_to_t = (self.arm_hist[:t]==i).sum() - n_to_t
        yloss = np.concatenate([self.yloss_arm_FSS[i][:n_to_t],
                        self.yloss_arm_GR[i][:g_to_t],
                        np.array([self.yloss[t,i]])], axis=0)
        X = np.concatenate([self.X_arm_FSS[i][:n_to_t],
                        self.X_arm_GR[i][:g_to_t],
                        self.X[t,:].reshape((1,self.k))], axis=0)
        return self.estimator(yloss, X)

    def do_forced(self):
        t = self.t
        # update parameters for arm pulled
        i = self.sampling[t]
        self.param_needs_update[i] = True

    def update_param(self):
        t = self.t
        for i in range(self.n_arms):
            if self.param_needs_update[i]:
                n_to_t = (self.sampling[:(t+1)]==i).sum()
                self.param[i] = self.estimator(self.yloss_arm_FSS[i][:n_to_t],
                    self.X_arm_FSS[i][:n_to_t,:])
                self.param_needs_update[i] = False

    def get_loss(self):
        t = self.t
        self.loss = self.yloss[np.arange(t), self.arm_hist[:t]]
        return self.loss

    def run(self):
        self.set_sample_procedure()
        self.reset()
        while self.t < self.N:
            self.next_action()

    def simulate(self, nsim=20, replace=True):
        self.sim_loss = []
        if self.run_oracle: self.sim_orac_loss = []
        for sim in range(nsim):
            self.resample(replace=replace)
            self.set_sample_procedure()
            self.reset()
            self.run()
            self.sim_loss.append(self.get_loss())
            if self.run_oracle:
                self.oracle()
                self.sim_orac_loss.append(self.oracle_loss)

    def plot_frac_incorrect(self, show_conf_bounds=True, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        points = np.arange(self.N)
        sim_losses = [self.sim_loss]
        if self.run_oracle: sim_losses.append(self.sim_orac_loss)
        for i in range(len(sim_losses)):
            cumloss = np.concatenate([make_col_vec(l.cumsum()) for l in sim_losses[i]],axis=1)
            avg_cumloss = cumloss.mean(axis=1)
            cumloss_upper = np.quantile(cumloss, .975, axis=1)
            cumloss_lower = np.quantile(cumloss, .025, axis=1)
            line_label = self.sample
            if i==1:
                line_label = 'oracle'
            ax.plot(points, avg_cumloss/(points+1), label=line_label)
            if show_conf_bounds:
                ax.fill_between(points,
                            cumloss_upper/(points+1),
                            cumloss_lower/(points+1),
                        alpha=0.2)
        ax.set_xlabel('Observations')
        ax.set_ylabel('Fraction Incorrect')
        return ax

bandit = FS_Bandit(yloss, X, sample='schedule', gr_filter=True, forced_samples=50, q=2, epsilon=.15, h=1)
bandit.run()
sim = False
if sim:
    bandit.simulate(nsim=10, replace=False)
    ax1 = bandit.plot_frac_incorrect()
    bandit.sample = 'epsilon'
    bandit.run_oracle = False
    bandit.simulate(nsim=10, replace=False)
    bandit.plot_frac_incorrect(ax=ax1)
    ax1.legend()
    plt.show()

