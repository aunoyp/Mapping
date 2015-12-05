# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import scipy as sp
import statsmodels.api as sm
from statsmodels.formula.api import ols

from experiment4 import Behavior


import matplotlib
import matplotlib.font_manager as font_manager
fontpath = '/Users/cjpeck/anaconda/lib/python3.4/site-packages/matplotlib/mpl-data/fonts/ttf/Helvetica.ttf'
prop = font_manager.FontProperties(fname=fontpath)
matplotlib.rcParams['font.family'] = prop.get_name()
matplotlib.rcParams['pdf.fonttype'] = 42


class BehavioralAnalyses(object):
    
    def __init__(self):
        self.dir_data = '/Users/cjpeck/Dropbox/Matlab/custom offline/mapping_py/mapping/data/'
        self.dir_figs = '/Users/cjpeck/Dropbox/Matlab/custom offline/mapping_py/mapping/figs/'
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
    
    def plot_performance(self, *args, title, labels, c):
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
        
    def df_for_statsmodel(self, *args, labels):        
        out = args[0]
        for arg in args[1:]:
            out = np.vstack((out, arg))
        return pd.DataFrame(np.transpose(out), columns=labels)
            
            
def print_and_write(f, line):
    print(line)
    if type(line) == pd.DataFrame:
        f.writelines(line.__str__())
    else:
        f.writelines(line)
            
    
if __name__ == '__main__':
    ba = BehavioralAnalyses()
    
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
        
                            