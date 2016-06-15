# -*- coding: utf-8 -*-

import itertools
from mapping.experiment import LoadData, Experiment
from mapping.session import Session
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import pandas as pd
import pickle
import scipy as sp
import statsmodels.api as sm
from statsmodels.formula.api import ols


class Behavior(object):
    ''' Behavior object:
    PURPOSE: get behavioral data from all sessions in order to perform
    population-level behavioral analyses
    '''
    def __init__(self, tFrame=[-500, 500]):
        self.directory = os.path.join(os.getenv('HOME'), 'GitHub/MappingData/')
        self.save_dir =  os.path.join(os.getenv('HOME'), 'GitHub/MappingData/')
        io = LoadData()
        self.files = np.array(list(io.experiment.files.keys()))
        self.nfiles = len(self.files)
        self.monkey = np.array([0 if file[0]=='t' else 1
                                for file in self.files])
        self.tFrame = tFrame
        self.nt = self.tFrame[1] - self.tFrame[0] + 1

    def extract_data(self):
        self.lick_rew = np.empty((self.nfiles, self.nt, 2))
        self.lick_rew_dir = np.empty((self.nfiles, self.nt, 2, 2))
        self.lick_rew_set = np.empty((self.nfiles, self.nt, 2, 2))
        self.hr_rew = np.empty((self.nfiles, 2))
        self.hr_rew_dir = np.empty((self.nfiles, 2, 2))
        self.hr_rew_set = np.empty((self.nfiles, 2, 2))
        self.rt_rew = np.empty((self.nfiles, 2))
        self.rt_rew_dir = np.empty((self.nfiles, 2, 2))
        self.rt_rew_set = np.empty((self.nfiles, 2, 2))

        self.lick_rew_set_dir = np.empty((self.nfiles, self.nt, 2, 2, 2))
        self.hr_rew_set_dir = np.empty((self.nfiles, 2, 2, 2))
        self.rt_rew_set_dir = np.empty((self.nfiles, 2, 2, 2))

        for iFile, file in enumerate(self.files):

            with open(self.directory + file + '.pickle', 'rb') as f:
                dat = pickle.load(f)

            # add fake reward times for the non-rew trials
            rewOn = np.array(dat.df['t_REWARD_ON'])
            isRew = np.array(dat.df['rew'].fillna(99), dtype=int)
            succOn = np.array(dat.df['t_SUCCESS'])
            delay = np.nanmean(rewOn[isRew==1] - succOn[isRew==1])
            rewOn[(isRew == 0) & ~np.isnan(succOn)] = succOn[(isRew == 0) & ~np.isnan(succOn)] + delay

            # other info
            cue_set = np.array(dat.df['cue_set'].fillna(99), dtype=int)
            cue_dir = np.array([1 if x < 0 else 0 if x > 0 else 99
                                for x in dat.df['CUE_X']], dtype=int)
            dat.df['rt'] = dat.df['t_FIX_OUT'] - dat.df['t_TARG_ON']

            # licking split by reward
            for irew in range(2):
                inds = np.logical_and(isRew == irew, ~np.isnan(rewOn))
                self.lick_rew[iFile, :, irew] = self.align_licks(dat.laser, inds, rewOn)
            # licking split by reward and direction
            for irew in range(2):
                for idir in range(2):
                    inds = np.logical_and(np.logical_and(isRew == irew, cue_dir == idir),
                                          ~np.isnan(rewOn))
                    self.lick_rew_dir[iFile, :, irew, idir] = self.align_licks(dat.laser, inds, rewOn)
            # licking split by reward and cue set
            for irew in range(2):
                for iset in range(2):
                    inds = np.logical_and(np.logical_and(isRew == irew, cue_set == iset),
                                          ~np.isnan(rewOn))
                    self.lick_rew_set[iFile, :, irew, iset] = self.align_licks(dat.laser, inds, rewOn)
            # licking split by reward, direction, cue set
            for irew in range(2):
                for iset in range(2):
                    for idir in range(2):
                        inds = np.logical_and(np.logical_and(isRew == irew, cue_set == iset, cue_dir == idir),
                                              ~np.isnan(rewOn))
                        self.lick_rew_set_dir[iFile, :, irew, iset, idir] = self.align_licks(dat.laser, inds, rewOn)

            # hit rate by reward
            for irew in range(2):
                hits = dat.df.ix[(isRew == irew) & (dat.df['hit'] | dat.df['miss']), 'hit']
                self.hr_rew[iFile, irew] = hits.sum() / len(hits)
            # hit rate by reward and direction
            for irew in range(2):
                for idir in range(2):
                    hits = dat.df.ix[(isRew == irew) & (cue_dir == idir) &
                                     (dat.df['hit'] | dat.df['miss']), 'hit']
                    self.hr_rew_dir[iFile, irew, idir] = hits.sum() / len(hits)
            # hit rate by reward and cue set
            for irew in range(2):
                for iset in range(2):
                    hits = dat.df.ix[(isRew == irew) & (cue_set == iset) &
                                     (dat.df['hit'] | dat.df['miss']), 'hit']
                    self.hr_rew_set[iFile, irew, iset] = hits.sum() / len(hits)
            # hit rate by reward, cue set, dir
            for irew in range(2):
                for iset in range(2):
                    for idir in range(2):
                        hits = dat.df.ix[(isRew == irew) & (cue_set == iset) & (cue_dir == idir) &
                                         (dat.df['hit'] | dat.df['miss']), 'hit']
                        self.hr_rew_set_dir[iFile, irew, iset, idir] = hits.sum() / len(hits)

            # reaction time by reward
            for irew in range(2):
                rt = dat.df.ix[(isRew == irew) & dat.df['hit'], 'rt']
                self.rt_rew[iFile, irew] = rt.mean(skipna=True)
            # reaction time by reward and direction
            for irew in range(2):
                for idir in range(2):
                    rt = dat.df.ix[(isRew == irew) & (cue_dir == idir) &
                                    dat.df['hit'], 'rt']
                    self.rt_rew_dir[iFile, irew, idir] = rt.mean(skipna=True)
            # reaction time by reward and cue set
            for irew in range(2):
                for iset in range(2):
                    rt = dat.df.ix[(isRew == irew) & (cue_set == iset) &
                                    dat.df['hit'], 'rt']
                    self.rt_rew_set[iFile, irew, iset] = rt.mean(skipna=True)
            # reaction time by reward, cue set, dir
            for irew in range(2):
                for iset in range(2):
                    for idir in range(2):
                        rt = dat.df.ix[(isRew == irew) & (cue_set == iset) & (cue_dir == idir) &
                                        dat.df['hit'], 'rt']
                        self.rt_rew_set_dir[iFile, irew, iset, idir] = rt.mean(skipna=True)

    def align_licks(self, laser, inds, etimes):
        licks = [laser[i] for i in range(len(laser)) if inds[i]]
        zero_inds = np.array(np.round(etimes[inds]*1000), dtype=int)
        out = np.empty((len(licks), self.nt))
        for i, lick in enumerate(licks):
            try:
                out[i,:] = lick[zero_inds[i] + self.tFrame[0] : zero_inds[i] + self.tFrame[1]+1]
            except:
                ipdb.set_trace()
        return np.nanmean(out, axis=0)

    def save_behavior(self):
        ''' Create a pickle of the behavior object '''
        fname = 'behavioral_data'
        print('saving:', self.save_dir + fname)
        with open(self.save_dir + fname + '.pickle', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


class BehavioralAnalyses(object):

    def __init__(self,  tFrameMean=[-250, 0]):
        self.dir_data = '../MappingData/'
        self.dir_figs = 'figs/'
        self.monkeys = ['Monkey T', 'Monkey S']
        self.directions = ['Ipsi.', 'Contra.']
        self.cuesets = ['Cue set 0', 'Cue set 1']
        self.tFrameMean = tFrameMean
        fname = 'behavioral_data'
        with open(self.dir_data + fname + '.pickle', 'rb') as f:
            self.dat = pickle.load(f)

    ### PLOT FORMATTING
    def line_with_shading(self, ax, x, y, c):
        yu = np.nanmean(y, axis=0)
        yerr = np.std(y, axis=0) / np.sqrt(y.shape[0])
        ax.plot(x, yu, c=c)
        ax.fill_between(x, yu-yerr, yu+yerr, facecolor=c, alpha=0.25)

    def format_licking_plot(self, ax, title):
        ax.set_xlabel('Time relative to reward onset (s)')
        ax.set_ylabel('Prop. time spent licking')
        ax.set_title(title)
        ax.set_xlim([x/1000 for x in self.dat.tFrame])
        ax.plot([0,0], plt.ylim(), c='k', linestyle='--')

    ### ANOVAS
    def _behavior_anova(self, data):
        """
        Data manipulation in prepartion for ANOVA
        """
        if data.ndim == 5:
            data = self.get_mean_across_t(data)
        neu_set0_ipsi_mean = data[:,0,0,0]
        rew_set0_ipsi_mean = data[:,1,0,0]
        neu_set0_cntr_mean = data[:,0,0,1]
        rew_set0_cntr_mean = data[:,1,0,1]
        neu_set1_ipsi_mean = data[:,0,1,0]
        rew_set1_ipsi_mean = data[:,1,1,0]
        neu_set1_cntr_mean = data[:,0,1,1]
        rew_set1_cntr_mean = data[:,1,1,1]

        y = np.hstack((
            neu_set0_ipsi_mean,
            rew_set0_ipsi_mean,
            neu_set0_cntr_mean,
            rew_set0_cntr_mean,
            neu_set1_ipsi_mean,
            rew_set1_ipsi_mean,
            neu_set1_cntr_mean,
            rew_set1_cntr_mean
        ))
        x_rew = np.hstack((
            np.zeros_like(neu_set0_ipsi_mean),
            np.ones_like(rew_set0_ipsi_mean),
            np.zeros_like(neu_set0_cntr_mean),
            np.ones_like(rew_set0_cntr_mean),
            np.zeros_like(neu_set1_ipsi_mean),
            np.ones_like(rew_set1_ipsi_mean),
            np.zeros_like(neu_set1_cntr_mean),
            np.ones_like(rew_set1_cntr_mean)
        ))
        x_set = np.hstack((
            np.zeros_like(neu_set0_ipsi_mean),
            np.zeros_like(rew_set0_ipsi_mean),
            np.zeros_like(neu_set0_cntr_mean),
            np.zeros_like(rew_set0_cntr_mean),
            np.ones_like(neu_set1_ipsi_mean),
            np.ones_like(rew_set1_ipsi_mean),
            np.ones_like(neu_set1_cntr_mean),
            np.ones_like(rew_set1_cntr_mean)
        ))
        x_dir = np.hstack((
            np.zeros_like(neu_set0_ipsi_mean),
            np.zeros_like(rew_set0_ipsi_mean),
            np.ones_like(neu_set0_cntr_mean),
            np.ones_like(rew_set0_cntr_mean),
            np.zeros_like(neu_set1_ipsi_mean),
            np.zeros_like(rew_set1_ipsi_mean),
            np.ones_like(neu_set1_cntr_mean),
            np.ones_like(rew_set1_cntr_mean)
        ))
        x_monkey = np.tile(self.dat.monkey, 8)

        return pd.DataFrame(
            data=np.c_[y, x_rew, x_set, x_dir, x_monkey],
            columns=['licking', 'rew', 'set', 'dir', 'monkey']
        )

    def licking_anova(self, max_interactions=None):
        """
        Do ANOVA on licking over all dimensions of the task.

        Features:
            - Reward
            - Cue set
            - Cue direction
            - Monkey

        Target:
            - Proportion time spent licking
        """
        df = self._behavior_anova(self.dat.lick_rew_set_dir)
        return anova(df, max_interactions=max_interactions)

    def hr_anova(self, max_interactions=None):
        """
        Do ANOVA on Hit Rate over all dimensions of the task.

        Features:
            - Reward
            - Cue set
            - Cue direction
            - Monkey

        Target:
            - Proportion time spent licking
        """
        df = self._behavior_anova(self.dat.hr_rew_set_dir)
        return anova(df, max_interactions=max_interactions)

    def rt_anova(self, max_interactions=None):
        """
        Do ANOVA on Reaction times over all dimensions of the task.

        Features:
            - Reward
            - Cue set
            - Cue direction
            - Monkey

        Target:
            - Proportion time spent licking
        """
        df = self._behavior_anova(self.dat.rt_rew_set_dir)
        return anova(df, max_interactions=max_interactions)


    ### PERFORMANCE PLOTTING
    def plot_licking(self):
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,4))

        t_plot = np.array(range(self.dat.tFrame[0], self.dat.tFrame[1]+1)) / 1000
        for i in range(3):
            if i == 0:
                inds = np.full_like(self.dat.monkey, fill_value=True, dtype=bool)
            else:
                inds = self.dat.monkey == i-1
            neu = self.dat.lick_rew[inds,:,0]
            rew = self.dat.lick_rew[inds,:,1]

            self.line_with_shading(ax=ax[i], x=t_plot, y=neu, c='r')
            self.line_with_shading(ax=ax[i], x=t_plot, y=rew, c='b')
            if i == 0:
                title = '{} - {} ms'.format(*self.tFrameMean)
            else:
                title = self.monkeys[i-1]
            self.format_licking_plot(ax=ax[i], title=title)

        fig.savefig(self.dir_figs + 'licking.pdf')

    def plot_hr(self):
        neu_mean = self.dat.hr_rew[:,0]
        rew_mean = self.dat.hr_rew[:,1]
        p = self.wilcoxon(neu_mean, rew_mean)
        title = 'Hit rate: p=%1.4f' %(p)
        self.plot_performance(neu_mean, rew_mean,
            title=title, labels=['Neutral', 'Reward'], c=['r', 'b'])

    def plot_rt(self):
        neu_mean = self.dat.rt_rew[:,0]
        rew_mean = self.dat.rt_rew[:,1]
        p = self.wilcoxon(neu_mean, rew_mean)
        title = 'Reaction time: p=%1.4f' %(p)
        self.plot_performance(neu_mean, rew_mean,
            title=title, labels=['Neutral', 'Reward'], c=['r', 'b'])

    def plot_performance(self, *args, **kwargs):
        title = kwargs['title']
        labels = kwargs['labels']
        c = kwargs['c']

        plt.figure()
        n = len(args)
        x = np.arange(n) + 0.5
        ymean, yerr = [], []
        for i, arg in enumerate(args):
            ymean.append(np.mean(arg))
            yerr.append(np.std(arg) / np.sqrt(len(arg)))
        plt.bar(x, ymean, width=0.6, color=c, yerr=yerr, align='center')

        plt.xticks(x, labels)
        plt.xlim([0, n])
        dtype = title[:title.find(':')]
        plt.title(title)
        plt.ylabel(dtype)
        plt.savefig(self.dir_figs + dtype + '.pdf')
        plt.show()

    ### WILCOXONS
    # licking
    def licking_wilcoxon(self):
        p, sign = [], []
        x0 = self.get_mean_across_t(self.dat.lick_rew[:,:,0])
        x1 = self.get_mean_across_t(self.dat.lick_rew[:,:,1])
        p.append(self.wilcoxon(x0, x1))
        sign.append(np.sign(np.mean(x1) - np.mean(x0)))
        return pd.DataFrame([p, sign], columns=['all'], index=['P', 'sign'])

    def licking_per_monkey(self):
        p, sign = [], []
        for imonk in range(2):
            x0 = self.get_mean_across_t(self.dat.lick_rew[self.dat.monkey==imonk,:,0])
            x1 = self.get_mean_across_t(self.dat.lick_rew[self.dat.monkey==imonk,:,1])
            p.append(self.wilcoxon(x0, x1))
            sign.append(np.sign(np.mean(x1) - np.mean(x0)))
        return pd.DataFrame([p, sign], columns=self.monkeys, index=['P', 'sign'])

    def licking_per_dir(self):
        p, sign = [], []
        for idir in range(2):
            x0 = self.get_mean_across_t(self.dat.lick_rew_dir[:,:,0,idir])
            x1 = self.get_mean_across_t(self.dat.lick_rew_dir[:,:,1,idir])
            p.append(self.wilcoxon(x0, x1))
            sign.append(np.sign(np.mean(x1) - np.mean(x0)))
        return pd.DataFrame([p, sign], columns=self.directions, index=['P', 'sign'])

    def licking_per_set(self):
        p, sign = [], []
        for iset in range(2):
            x0 = self.get_mean_across_t(self.dat.lick_rew_set[:,:,0,iset])
            x1 = self.get_mean_across_t(self.dat.lick_rew_set[:,:,1,iset])
            p.append(self.wilcoxon(x0, x1))
            sign.append(np.sign(np.mean(x1) - np.mean(x0)))
        return pd.DataFrame([p, sign], columns=self.cuesets, index=['P', 'sign'])

    # hit rate
    def hr_wilcoxon(self):
        p, sign = [], []
        x0 = self.dat.hr_rew[:,0]
        x1 = self.dat.hr_rew[:,1]
        p.append(self.wilcoxon(x0, x1))
        sign.append(np.sign(np.mean(x1) - np.mean(x0)))
        return pd.DataFrame([p, sign], columns=['all'], index=['P', 'sign'])

    def hr_per_monkey(self):
        p, sign = [], []
        for imonk in range(2):
            x0 = self.dat.hr_rew[self.dat.monkey==imonk, 0]
            x1 = self.dat.hr_rew[self.dat.monkey==imonk, 1]
            p.append(self.wilcoxon(x0, x1))
            sign.append(np.sign(np.mean(x1) - np.mean(x0)))
        return pd.DataFrame([p, sign], columns=self.monkeys, index=['P', 'sign'])

    def hr_per_dir(self):
        p, sign = [], []
        for idir in range(2):
            x0 = self.dat.hr_rew_dir[:, 0, idir]
            x1 = self.dat.hr_rew_dir[:, 1, idir]
            p.append(self.wilcoxon(x0, x1))
            sign.append(np.sign(np.mean(x1) - np.mean(x0)))
        return pd.DataFrame([p, sign], columns=self.directions, index=['P', 'sign'])

    def hr_per_set(self):
        p, sign = [], []
        for iset in range(2):
            x0 = self.dat.hr_rew_set[:, 0, iset]
            x1 = self.dat.hr_rew_set[:, 1, iset]
            p.append(self.wilcoxon(x0, x1))
            sign.append(np.sign(np.mean(x1) - np.mean(x0)))
        return pd.DataFrame([p, sign], columns=self.cuesets, index=['P', 'sign'])

    # reaction time
    def rt_wilcoxon(self):
        p, sign = [], []
        x0 = self.dat.rt_rew[:,0]
        x1 = self.dat.rt_rew[:,1]
        p.append(self.wilcoxon(x0, x1))
        sign.append(np.sign(np.mean(x1) - np.mean(x0)))
        return pd.DataFrame([p, sign], columns=['all'], index=['P', 'sign'])

    def rt_per_monkey(self):
        p, sign = [], []
        for imonk in range(2):
            x0 = self.dat.rt_rew[self.dat.monkey==imonk, 0]
            x1 = self.dat.rt_rew[self.dat.monkey==imonk, 1]
            p.append(self.wilcoxon(x0, x1))
            sign.append(np.sign(np.mean(x1) - np.mean(x0)))
        return pd.DataFrame([p, sign], columns=self.monkeys, index=['P', 'sign'])

    def rt_per_dir(self):
        p, sign = [], []
        for idir in range(2):
            x0 = self.dat.rt_rew_dir[:, 0, idir]
            x1 = self.dat.rt_rew_dir[:, 1, idir]
            p.append(self.wilcoxon(x0, x1))
            sign.append(np.sign(np.mean(x1) - np.mean(x0)))
        return pd.DataFrame([p, sign], columns=self.directions, index=['P', 'sign'])

    def rt_per_set(self):
        p, sign = [], []
        for iset in range(2):
            x0 = self.dat.rt_rew_set[:, 0, iset]
            x1 = self.dat.rt_rew_set[:, 1, iset]
            p.append(self.wilcoxon(x0, x1))
            sign.append(np.sign(np.mean(x1) - np.mean(x0)))
        return pd.DataFrame([p, sign], columns=self.cuesets, index=['P', 'sign'])

    ### STATS
    def wilcoxon(self, x0, x1):
        """
        Wilcoxon test to use.

        sp.stats.wilcoxon: paired data
        sp.stats.mannwhitneyu: 2-sample (unpaired data)
        """
        _, p = sp.stats.wilcoxon(x0, x1)
        return p

    ### AUXILLARY
    def get_mean_across_t(self, x):
        t_all = np.arange(self.dat.tFrame[0], self.dat.tFrame[1]+1)
        t_mean = np.logical_and(t_all >= self.tFrameMean[0],
                                t_all <= self.tFrameMean[1])
        return np.mean(x[:,t_mean], axis=1)



def anova(df, max_interactions=None):
    """
    Generalized ANOVA functions.

    Assumed 0th column of dataframe is the target.
    """

    cols = list(df.columns)
    if not max_interactions:
        max_interactions = df.shape[1] - 1
    formula = '{} ~ '.format(cols[0])
    cols.pop(0)

    for choose in range(max_interactions):
        combs = itertools.combinations(cols, choose + 1)
        for comb in combs:
            for i, col in enumerate(comb):
                if i == 0:
                    formula += 'C({})'.format(col)
                else:
                    formula += ':C({})'.format(col)
                if (i == len(comb)-1):
                    formula += ' + '
    formula = formula[:-3]

    lm = ols(formula, df).fit()
    return sm.stats.anova_lm(lm)

def print_and_write(f, line):
    print(line)
    if type(line) == pd.DataFrame:
        f.writelines(line.__str__())
    else:
        f.writelines(line)

def figs_and_text(ba):
    with open(ba.dir_figs + 'behavioral_results.txt', 'w') as f:
        ba.plot_licking()
        print_and_write(f, '\n**** LICKING: ACROSS MONKEY\n')
        print_and_write(f, ba.licking_across_monkey())
        print_and_write(f, '\n**** LICKING: ACROSS DIRECTION\n')
        print_and_write(f, ba.licking_across_dir())

        ba.plot_hr()
        #print_and_write(f, '\n*** HIT RATE: ACROSS MONKEY\n')
        #print_and_write(f, ba.hr_across_monkey())
        #print_and_write(f, '\n**** HIT RATE: ACROSS DIRECTION\n')
        #print_and_write(f, ba.hr_across_dir())
        print_and_write(f, '\n*** HIT RATE: PER MONKEY\n')
        print_and_write(f, ba.hr_per_monkey())
        print_and_write(f, '\n*** HIT RATE: PER DIRECTION\n')
        print_and_write(f, ba.hr_per_dir())
        print_and_write(f, '\n*** HIT RATE: PER SET\n')
        print_and_write(f, ba.hr_per_set())

        ba.plot_rt()
        #print_and_write(f, '\n**** REACTION TIME: ACROSS MONKEY\n')
        #print_and_write(f, ba.rt_across_monkey())
        #print_and_write(f, '\n**** REACTION TIME: ACROSS DIRECTION\n')
        #print_and_write(f, ba.rt_across_dir())
        print_and_write(f, '\n*** REACTION TIME: PER MONKEY\n')
        print_and_write(f, ba.rt_per_monkey())
        print_and_write(f, '\n*** REACTION TIME: PER DIRECTION\n')
        print_and_write(f, ba.rt_per_dir())
        print_and_write(f, '\n*** REACTION TIME: PER SET\n')
        print_and_write(f, ba.rt_per_set())

if __name__ == '__main__':

    # b = Behavior()
    # b.extract_data()
    # b.save_behavior()

    ba = BehavioralAnalyses()
    # ba.plot_licking()
    print(ba.licking_anova())
    print(ba.hr_anova())
    print(ba.rt_anova())
