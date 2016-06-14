# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import pandas as pd
import pickle
import scipy as sp
import statsmodels.api as sm
from statsmodels.formula.api import ols

#import matplotlib.font_manager as font_manager
#fontpath = '/Users/cjpeck/anaconda/lib/python3.4/site-packages/matplotlib/mpl-data/fonts/ttf/Helvetica.ttf'
#prop = font_manager.FontProperties(fname=fontpath)
#matplotlib.rcParams['font.family'] = prop.get_name()
#matplotlib.rcParams['pdf.fonttype'] = 42


class Behavior(object):
    ''' Behavior object:
    PURPOSE: get behavioral data from all sessions in order to perform
    population-level behavioral analyses
    '''
    def __init__(self, tFrame=[-500, 500]):
        self.directory = '/Users/syi115/GitHub/MappingData/'
        self.save_dir = '/Users/syi115/GitHub/MappingData/'
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

    def __init__(self):
        self.dir_data = '../MappingData/'
        self.dir_figs = 'figs/'
        self.monkeys = ['Tom', 'Spaghetti']
        self.directions = ['Ipsi.', 'Contra.']
        self.cuesets = ['Cue set 0', 'Cue set 1']
        self.tFrameMean = [-250, 0]
        fname = 'behavioral_data'
        with open(self.dir_data + fname + '.pickle', 'rb') as f:
            self.dat = pickle.load(f)

    ### plotting
    def plot_licking(self):
        plt.figure()
        t_plot = np.array(range(self.dat.tFrame[0], self.dat.tFrame[1]+1)) / 1000
        neu = self.dat.lick_rew[:,:,0]
        rew = self.dat.lick_rew[:,:,1]
        self.line_with_shading(t_plot, neu, 'r')
        self.line_with_shading(t_plot, rew, 'b')
        neu_mean = self.get_mean_across_t(neu)
        rew_mean = self.get_mean_across_t(rew)
        p = self.wilcoxon(neu_mean, rew_mean)
        title = '%d - %d ms: p=%1.4f' %(self.tFrameMean[0], self.tFrameMean[1], p)
        self.format_plot(title)
        plt.savefig(self.dir_figs + 'licking.pdf')
        plt.show()

    ### PLOT FORMATTING
    def line_with_shading(self, x, y, c):
        yu = np.nanmean(y, axis=0)
        yerr = np.std(y, axis=0) / np.sqrt(y.shape[0])
        plt.plot(x, yu, c=c)
        plt.fill_between(x, yu-yerr, yu+yerr, facecolor=c, alpha=0.25)

    def format_plot(self, title):
        plt.xlabel('Time relative to reward onset (s)')
        plt.ylabel('Prop. time spent licking')
        plt.title(title)
        plt.xlim([x/1000 for x in self.dat.tFrame])
        plt.plot([0,0], plt.ylim(), c='k', linestyle='--')

    ### LICKING WRAPPERS
    def licking_across_monkey(self):
        neu_mean = self.get_mean_across_t(self.dat.lick_rew[:,:,0])
        rew_mean = self.get_mean_across_t(self.dat.lick_rew[:,:,1])
        y = np.hstack((neu_mean, rew_mean))
        x1 = np.hstack((np.zeros_like(neu_mean), np.ones_like(rew_mean)))
        x2 = np.tile(self.dat.monkey, 2)
        df = self.df_for_statsmodel(y, x1, x2, labels=['licking', 'rew', 'monkey'])
        return self.anova_2way(df)

    def licking_across_dir(self):
        neu_ipsi_mean = self.get_mean_across_t(self.dat.lick_rew_dir[:,:,0,0])
        rew_ipsi_mean = self.get_mean_across_t(self.dat.lick_rew_dir[:,:,1,0])
        neu_cntr_mean = self.get_mean_across_t(self.dat.lick_rew_dir[:,:,0,1])
        rew_cntr_mean = self.get_mean_across_t(self.dat.lick_rew_dir[:,:,1,1])
        y, x1, x2 = self.arrays_for_anova_2way(neu_ipsi_mean, rew_ipsi_mean,
                                               neu_cntr_mean, rew_cntr_mean)
        df = self.df_for_statsmodel(y, x1, x2, labels=['licking', 'rew', 'dir'])
        return self.anova_2way(df)

    def licking_anova(self):
        neu_ipsi_mean = self.get_mean_across_t(self.dat.lick_rew_dir[:,:,0,0])
        rew_ipsi_mean = self.get_mean_across_t(self.dat.lick_rew_dir[:,:,1,0])
        neu_cntr_mean = self.get_mean_across_t(self.dat.lick_rew_dir[:,:,0,1])
        rew_cntr_mean = self.get_mean_across_t(self.dat.lick_rew_dir[:,:,1,1])
        y, x_rew, x_dir = self.arrays_for_anova_2way(neu_ipsi_mean, rew_ipsi_mean,
                                               neu_cntr_mean, rew_cntr_mean)
        x_monkey = np.tile(self.dat.monkey, 4)
        df = self.df_for_statsmodel(y, x_rew, x_dir, x_monkey, labels=['licking', 'rew', 'dir', 'monkey'])
        return self.anova_3way(df)

    ### PERFORMANCE PLOTTING
    def plot_hr(self):
        neu_mean = self.dat.hr_rew[:,0]
        rew_mean = self.dat.hr_rew[:,1]
        p = self.wilcoxon(neu_mean, rew_mean)
        title = 'Hit rate: p=%1.4f' %(p)
        self.plot_performance(neu_mean, rew_mean, title=title, labels=['Neutral', 'Reward'], c=['r', 'b'])

    def plot_rt(self):
        neu_mean = self.dat.rt_rew[:,0]
        rew_mean = self.dat.rt_rew[:,1]
        p = self.wilcoxon(neu_mean, rew_mean)
        title = 'Reaction time: p=%1.4f' %(p)
        self.plot_performance(neu_mean, rew_mean, title=title, labels=['Neutral', 'Reward'], c=['r', 'b'])

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

    ### 2-ways ANOVAs to look at other effects
    def hr_across_monkey(self):
        return self.perf_across_monkey(self.dat.hr_rew[:,0], self.dat.hr_rew[:,1], 'HR')

    def rt_across_monkey(self):
        return self.perf_across_monkey(self.dat.rt_rew[:,0], self.dat.rt_rew[:,1], 'RT')

    def perf_across_monkey(self, y0, y1, ylabel):
        y = np.hstack((y0, y1))
        x1 = np.hstack((np.zeros_like(y0), np.ones_like(y1)))
        x2 = np.tile(self.dat.monkey, 2)
        df = self.df_for_statsmodel(y, x1, x2, labels=[ylabel, 'rew', 'monkey'])
        return self.anova_2way(df)

    def hr_across_dir(self):
        neu_ipsi_mean = self.dat.hr_rew_dir[:,0,0]
        rew_ipsi_mean = self.dat.hr_rew_dir[:,1,0]
        neu_cntr_mean = self.dat.hr_rew_dir[:,0,1]
        rew_cntr_mean = self.dat.hr_rew_dir[:,1,1]
        y, x1, x2 = self.arrays_for_anova_2way(neu_ipsi_mean, rew_ipsi_mean,
                                               neu_cntr_mean, rew_cntr_mean)
        df = self.df_for_statsmodel(y, x1, x2, labels=['HR', 'rew', 'dir'])
        return self.anova_2way(df)

    def rt_across_dir(self):
        neu_ipsi_mean = self.dat.rt_rew_dir[:,0,0]
        rew_ipsi_mean = self.dat.rt_rew_dir[:,1,0]
        neu_cntr_mean = self.dat.rt_rew_dir[:,0,1]
        rew_cntr_mean = self.dat.rt_rew_dir[:,1,1]
        y, x1, x2 = self.arrays_for_anova_2way(neu_ipsi_mean, rew_ipsi_mean,
                                               neu_cntr_mean, rew_cntr_mean)
        df = self.df_for_statsmodel(y, x1, x2, labels=['RT', 'rew', 'dir'])
        return self.anova_2way(df)

    ### wilcoxon to verify reward effect in each condtion
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
        _, p = sp.stats.wilcoxon(x0, x1)
        return p

    def anova_2way(self, df):
        formula = '%s ~ C(%s) + C(%s) + C(%s):C(%s)' %(df.keys()[0], df.keys()[1], df.keys()[2], df.keys()[1], df.keys()[2])
        lm = ols(formula, df).fit()
        return sm.stats.anova_lm(lm)

    def anova_3way(self, df):
        cols = list(df.columns)
        formula = '{} ~ C({}) + C({}) + C({}) + C({}):C({}) + C({}):C({}) + C({}):C({}) + C({}):C({}):C({})'.format(
            *(cols + cols[1:3] + [cols[1]] + [cols[3]] + cols[2:] + cols[1:])
        )
        lm = ols(formula, df).fit()
        return sm.stats.anova_lm(lm)

    ### AUXILLARY
    def get_mean_across_t(self, x):
        t_all = np.arange(self.dat.tFrame[0], self.dat.tFrame[1]+1)
        t_mean = np.logical_and(t_all >= self.tFrameMean[0],
                                t_all <= self.tFrameMean[1])
        return np.mean(x[:,t_mean], axis=1)

    def arrays_for_anova_2way(self, y0a, y1a, y0b, y1b):
        y = np.hstack((y0a, y1a, y0b, y1b))
        x1 = np.hstack((np.zeros_like(y0a), np.ones_like(y1a),
                        np.zeros_like(y0b), np.ones_like(y1b)))
        x2 = np.hstack((np.zeros_like(y0a), np.zeros_like(y1a),
                        np.ones_like(y0b), np.ones_like(y1b)))
        return y, x1, x2

    def df_for_statsmodel(self, *args, **kwargs):
        out = args[0]
        for arg in args[1:]:
            out = np.vstack((out, arg))
        return pd.DataFrame(np.transpose(out), columns=kwargs['labels'])


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

    #b = Behavior()
    #b.extract_data()
    #b.save_behavior()

    ba = BehavioralAnalyses()
    # ba.plot_licking()
    ba.licking_anova()


