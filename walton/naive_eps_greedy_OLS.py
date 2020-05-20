#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

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
    return np.linalg.lstsq(X, y, rcond=None)

# choose arm based on OLS
def OLS_arm(OLS_params, x):
    losses = []
    for B in OLS_params:
        losses.append(np.asmatrix(x @ B).T)
    losses = np.concatenate(losses, axis=1)
    return np.asarray(losses.argmin(axis=1)).flatten()

# empirical losses
yloss = np.zeros((N,n_arms))
for y_pull in arms:
    yloss[:,y_pull] = loss01(y_pull,y)

class NaiveEpsGreedyBandit:

    def __init__(self, yloss, X, forced_samples=None, epsilon=.1, seed=None):
        self.yloss0 = yloss
        self.X0 = X
        self.N = X.shape[0]
        self.k = X.shape[1]
        self.n_arms = yloss.shape[1]
        self.epsilon = epsilon
        # default number of initial forced samples is 1/4 of the data
        # divided equally among the arms
        if forced_samples is None:
            forced_samples = int(N * .25/ n_arms)
        self.forced_samples = forced_samples
        self.reset_data()
        self.set_seed(seed)
        self.oracle()

    def set_seed(self, seed=None):
        self.seed = seed
        if self.seed is not None:
            np.random.seed(seed)

    def set_sample_procedure(self):
        block = np.tile(np.arange(self.n_arms),
                        self.forced_samples)
        greedy = np.random.binomial(1, self.epsilon, self.N - self.forced_samples*self.n_arms)
        nforced = greedy.sum()
        greedy = greedy - 1
        greedy[greedy == 0] = np.random.randint(self.n_arms, size=nforced)
        self.sampling = np.concatenate([block, greedy])
        self.arm_hist = self.sampling.copy()

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
        self.OLS_param = []
        self.OLS_needs_update = []
        for i in range(self.n_arms):
            # features and losses corresponding to each arm
            self.X_arm_FSS.append(self.X[self.sampling==i,:])
            self.yloss_arm_FSS.append(self.yloss[self.sampling==i,i])
            # greedy sample preallocate all space
            self.X_arm_GR.append(np.zeros((self.N - self.forced_samples*self.n_arms, self.k)))
            self.yloss_arm_GR.append(np.zeros(self.N - self.forced_samples*self.n_arms))
            self.OLS_param.append(np.zeros((self.k,)))
            self.OLS_needs_update.append(True)
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
            mod = fastOLS(self.yloss[:,i], self.X)[0]
            oracle_params.append(mod)
        self.oracle_params = oracle_params
        # oracle arm choices
        self.oracle_arm = OLS_arm(self.oracle_params, self.X).flatten()
        # oracle loss
        self.oracle_loss = self.yloss[np.arange(self.N),self.oracle_arm]
        self.oracle_cumloss = self.oracle_loss.cumsum()

    def next_eps_greedy(self):
        arm = self.sampling[self.t]
        # greedy is denoted by arm = -1
        if arm==-1:
            self.do_greedy()
        else:
            self.do_forced()
        self.t += 1

    def do_greedy(self):
        self.update_OLS()
        t = self.t
        # take the greedy action
        arm_current = OLS_arm(
            self.OLS_param, self.X[t,:])[0]
        self.arm_hist[t] = arm_current
        greedy_ind = ((self.arm_hist[:(t+1)]==arm_current).sum() -
                (self.sampling[:(t+1)]==arm_current).sum())
        self.X_arm_GR[arm_current][greedy_ind,:] = self.X[greedy_ind,:].reshape((1,self.k))
        self.yloss_arm_GR[arm_current][greedy_ind] = self.yloss[greedy_ind,arm_current]

    def do_forced(self):
        t = self.t
        # update OLS parameters for arm pulled
        i = self.sampling[t]
        self.OLS_needs_update[i] = True

    def update_OLS(self):
        t = self.t
        for i in range(self.n_arms):
            if self.OLS_needs_update[i]:
                n_to_t = (self.sampling[:(t+1)]==i).sum()
                self.OLS_param[i] = fastOLS(self.yloss_arm_FSS[i][:n_to_t],
                    self.X_arm_FSS[i][:n_to_t,:])[0]
                self.OLS_needs_update[i] = False

    def get_loss(self):
        t = self.t
        self.loss = self.yloss[np.arange(t), self.arm_hist[:t]]
        return self.loss

    def run(self):
        self.set_sample_procedure()
        self.reset()
        while self.t < self.N:
            self.next_eps_greedy()

    def simulate(self, nsim=20, replace=True):
        self.sim_loss = []
        for sim in range(nsim):
            self.resample(replace=replace)
            self.set_sample_procedure()
            self.reset()
            self.run()
            self.sim_loss.append(self.get_loss())

    def plot_frac_incorrect(self, show_conf_bounds=True):
        fig, ax = plt.subplots()
        points = np.arange(self.N)
        ax.plot(points, self.oracle_cumloss/(points+1))
        ax.plot(points, self.cumloss_mean/(points+1))
        ax.set_xlabel('Observations')
        ax.set_ylabel('Fraction Incorrect')
        if show_conf_bounds:
            ax.fill_between(points,
                        (self.cumloss_mean - 1.96*self.cumloss_std)/(points+1),
                        (self.cumloss_mean + 1.96*self.cumloss_std)/(points+1),
                       alpha=0.2)

bandit = NaiveEpsGreedyBandit(yloss, X, forced_samples=20, epsilon=.2)
bandit.run()
#bandit.simulate(nsim=20, replace=False)
#bandit.plot_frac_incorrect()
#plt.show()

