# -*- coding: utf-8 -*-

from collections import OrderedDict
from mapping.session import Session
from mapping.neuron import Neuron
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import os
import pickle
import scipy as sp
import scipy.stats as stats
import pandas as pd
from scipy.optimize import curve_fit
import scipy.io
import glob
import numpy as np
import copy


class Experiment(object):
    '''Experiment object: Maintains dictionary of experiment sessions and
    corresponding neurons.'''
    def __init__(self, OVERWRITE=False):
        ''' Initialize dictionary'''
        self.OVERWRITE = OVERWRITE
        if not self.OVERWRITE:
            self.load_experiment()
        else:
            self.files = OrderedDict()

        self.directory = os.path.join(os.sep, 'Volumes/AUNOY_SALZ/')

    def add_session(self, f, fname, cells):
        '''Add a session to the dictionary and create the 'Session' object for
        that sessions.

        If session is already in the dictionary, do nothing unless
        self.OVERWRITE = True
        '''
        if len(cells)==0:
            print(fname, ' has no cells?')
            return
        if not fname in self.files or self.OVERWRITE:
            self.files[fname] = cells
            session = Session(f, fname, cells, self.OVERWRITE)
            session.save_session()

    def save_experiment(self):
        '''Save experiment file'''
        print('saving: mapping_exp')
        with open(self.directory + 'mapping_exp.pickle', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load_experiment(self):
        '''Save experiment file'''
        print('loading: mapping_exp')
        with open(self.directory + 'mapping_exp.pickle', 'rb') as f:
            dat = pickle.load(f)
            self.files = dat.files

class LoadData(object):
    ''' LoadData object:
    PURPOSE: Interface for loading saved & processed data files for Session
    and Neuron Objects
    FUNCTIONALITY: Just loads them
    '''
    def __init__(self):
        self.directory = os.path.join(os.sep, 'Volumes/AUNOY_SALZ/') 
        with open(self.directory + 'mapping_exp.pickle', 'rb') as f:
            self.experiment = pickle.load(f)

    def load_session(self, fname):

        while not fname in self.experiment.files:
            tmp_list = []
            for i, key in enumerate(self.experiment.files):
                print(i, key)
                tmp_list.append(key)
            ind = input('Which session?:  ')
            fname = tmp_list[int(ind)]

        with open(self.directory + fname + '.pickle', 'rb') as f:
            session = pickle.load(f)
        return session

    def load_neuron(self, filename, cellname):

        while not filename in self.experiment.files:
            tmp_list = []
            for i, key in enumerate(self.experiment.files):
                print(i, key)
                tmp_list.append(key)
            ind = input('Which session?:  ')
            filename = tmp_list[int(ind)]

        while not cellname in self.experiment.files[filename]:
            tmp_list = []
            for i, key in enumerate(self.experiment.files[filename]):
                print(i, key)
                tmp_list.append(key)
            ind = input('Which neuron?:  ')
            cellname = tmp_list[int(ind)]

        fname = filename + '_' + cellname
        with open(self.directory + fname + '.pickle', 'rb') as f:
            neuron = pickle.load(f)
        return neuron

def expo_fun(x, lam):
    return lam*np.exp((-lam)*x)


def demoOld(elecName=None, one_example=True):

    io = LoadData()
    if one_example == True:
        demo_neurons = {'tn_map_123014': ['elec11U']}
    else:
        demo_neurons = {'tn_map_120314': [elecName],
                        'tn_map_120414': [elecName],
                        'tn_map_120514': [elecName],
                        'tn_map_120814': [elecName],
                        'tn_map_120914': [elecName],
                        'tn_map_121114': [elecName],
                        'tn_map_121214': [elecName],
                        'tn_map_121514': [elecName],
                        'tn_map_121614': [elecName],
                        'tn_map_121714': [elecName],
                        'tn_map_122914': [elecName],
                        'tn_map_123014': [elecName]}
    for file in demo_neurons:
        for cell in demo_neurons[file]:

            # load neuron object
            print('Loading neuron', file, cell)
            neuron = io.load_neuron(file, cell)

            # X/Y information for stimuli in mapping experiment
            neuron.get_xy()
            neuron.get_xy_grid()
            neuron.get_xy_plot_grid()
            

            fileShort ='/Users/aunoyp/Documents/Salzman_Lab/Thesis/Figs/'+ file + '_' + elecName
            fRew = fileShort + 'rew.pdf'
            # Firing sorted by reward condition (ignoring spatial location)
            neuron.smooth_firing_rates()
            neuron.psth_rew(pdf=fRew)

            fmap = fileShort + 'map.pdf'
            # Firing rates for each location in experiment
            neuron.get_frmean_by_loc((.1,.5))
            neuron.get_fr_by_loc()
            neuron.psth_map(pdf=fmap)

            # Use firing rates to determine initial guess paramaters for 2d
            # gaussian fit
            neuron.get_initial_params()

            # Fit guassian function with 'basinhopping' algorithm in attempt
            # find a global minimum in this paramater space
            neuron.fit_gaussian()

            fgauss = fileShort + 'gauss.pdf'
            # Get high-res guassian for plotting based on the fitted parameters
            g_nr = neuron.get_gauss(neuron.x_plot_grid, neuron.y_plot_grid, 0,
                                    neuron.betas)
            g_rw = neuron.get_gauss(neuron.x_plot_grid, neuron.y_plot_grid, 1,
                                    neuron.betas)
            neuron.plot_gaussian([g_nr, g_rw], neuron.frmean_space, pdf=fgauss)

def demoDDD():
    io = LoadData()
    os.chdir("/Volumes/AUNOY_SALZ")
    for file in glob.glob("*elec*.pickle"):
        filepart = file.split("_")
        expName = filepart[0] + "_" + filepart[1] + "_" +filepart[2]
        nname = filepart[3].split(".")
        neuron = io.load_neuron(expName, nname[0])
        neuron.get_xy()
        neuron.get_xy_grid()
        neuron.get_xy_plot_grid()
        x = neuron.x
        y = neuron.y
        neuron.smooth_firing_rates()
        rewOff= np.array(np.zeros((neuron.ntrials, 20))) 
        #rewOn = np.array(np.zeros((5,5)))
        fr0 = np.nanmean(neuron.fr_smooth[np.array(neuron.df['rew']==0),:], 0)
        print(neuron.df.shape)
        print("\n\n\n")

def ROC():
# ROC Heatmap
    
    from sklearn.metrics import roc_curve, auc
    io = LoadData()
    os.chdir("/Volumes/AUNOY_SALZ")
    x = []
    y = []
    filepath = "/Volumes/AUNOY_SALZ/figs/"

    for file in glob.glob("*_map_*elec*.pickle"):
        filepart = file.split("_")
        expName = filepart[0] + "_" + filepart[1] + "_" +filepart[2]
        nname = filepart[3].split(".")
        neuron = io.load_neuron(expName, nname[0])
        
        roc_auc_hmap = np.full((5,5), np.nan)
        neuron.get_xy()
        neuron.get_xy_grid()
        neuron.get_xy_plot_grid()
        neuron.smooth_firing_rates()

        ''' 
        So the code that I will have to write will return the indices that I want
        for each trial, and then I will compute the mean firing rate for that
        trial in the cue onset window, which is 100 - 500 milliseconds. I need
        to separate these into two arrays. We will first do the location -12,-12
            fr_dist: array of len(trials of rew for -12,-12) with parameters,
            one is the mean firing rate and second is reward value.
            
        '''

        for ix in range(len(neuron.x)):
            for iy in range(len(neuron.y)):
                
                inds = neuron.get_xy_rew_inds(neuron.x[ix], neuron.y[iy], 0)

                t = neuron.get_mean_t((0.1, 0.5))
                mean_trial_fr = np.nanmean(neuron.fr_smooth[np.ix_(inds, t)], 1)
                mean_trial_fr = mean_trial_fr[~np.isnan(mean_trial_fr)]
                fr_noR = np.zeros((len(mean_trial_fr), 2))
                fr_noR[:, 0] = mean_trial_fr
                
                
                inds = neuron.get_xy_rew_inds(neuron.x[ix], neuron.y[iy], 1)
                mean_trial_frR = np.nanmean(neuron.fr_smooth[np.ix_(inds, t)], 1)
                mean_trial_frR = mean_trial_frR[~np.isnan(mean_trial_frR)]
                fr_R = np.ones((len(mean_trial_frR), 2))
                fr_R[:, 0] = mean_trial_frR

                fr_dist = np.concatenate((fr_noR, fr_R), axis=0)
                fr_bny = fr_dist[:, 1]
                fr_bny = fr_bny.astype(int)
                

                # Compute ROC curve and ROC area for each class
                fpr, tpr, thresholds = roc_curve(fr_bny, fr_dist[:,0])
                roc_auc = auc(fpr, tpr)
                roc_auc_hmap[len(neuron.y)-1-iy, ix] = roc_auc
                    
        fig, ax = plt.subplots()

        im, cbar = heatmap(roc_auc_hmap, np.flip(y,0), x,  ax=ax, cmap='PiYG', cbarlabel="ROC AUC", vmin= 0, vmax=1)
        texts = annotate_heatmap(im, valfmt="{x:.3g}")

        fig.tight_layout()
        pdf = filepath+expName + nname[0] + "_cue_roc_heatmap.pdf"
        plt.savefig(pdf, dpi=150)
        plt.close()

def Recursive_GLM():

    import math
    import statsmodels.api as sm
    from scipy import stats

    io = LoadData()
    os.chdir("/Volumes/AUNOY_SALZ")
    filepath = "/Volumes/AUNOY_SALZ/figs/"
    f = open("GLM_Recursive_XY.txt", "w")

    for file in glob.glob("*map_*elec*.pickle"):
        filepart = file.split("_")
        expName = filepart[0] + "_" + filepart[1] + "_" +filepart[2]
        nname = filepart[3].split(".")
        print(expName + " " + nname[0])
        neuron = io.load_neuron(expName, nname[0])
        #if(expName == 'sn_map_051215' and nname[0]== 'elec7b'):
        #    continue
        #if(expName == 'tn_map_120514' and nname[0] == 'elec19a'):
        #    continue
        #if(expName == 'tn_map_120814' and nname[0] == 'elec10a'):
        #    continue
        #if(expName == 'tn_map_121514' and nname[0] == 'elec14a'):
        #    continue
        #if(expName == 'sn_map_051515' and nname[0] == 'elec16a'):
        #    continue
        #if(expName == 'sn_map_051515' and nname[0] == 'elec16a'):
        #    continue
        #if(expName == 'sn_map_051915' and nname[0] == 'elec29b'):
        #    continue
        #if(expName == 'tn_map_120814' and nname[0] == 'elec6c'):
        #    continue

        neuron.get_xy()
        neuron.get_xy_grid()
        neuron.get_xy_plot_grid()

        x = neuron.x
        y = neuron.y
        neuron.smooth_firing_rates()

        '''
        I need to create a dataframe that only has the r, theta, cue, r-cue,
        theta-cue, r-theta-cue, and fr data.

        '''

        only_good = neuron.df.loc[neuron.df['hit'] == True]

        term = ['CUE_TYPE', 'CUE_X', 'CUE_Y', 'trial_num']
        terms = only_good.loc[:, term]

        terms['r^2'] = terms['CUE_X']**2 + terms['CUE_Y']**2
        terms['theta'] = np.arctan(terms['CUE_Y'] / terms['CUE_X'])

        terms['theta'].fillna((math.pi/2), inplace=True)



        neuron.fr_smooth

        '''

        I want to calculate the mean fr for the cue, trace, and target 
        interval. To do this I need to get the average for one trial.
        In terms, I have a list of good trials that I can use.
        As I iterate through, these will have an x condition, y condition
        and a firing rate. What I want to do is access the fr_smooth
        array and get the average firing rate across.  So I need to iterate
        through the good conditions, extract the trial number and cross
        reference to the original df.

        Will wait to look at epoch later.
        '''


        cue_fr = np.zeros(len(terms))
        trace_fr = np.zeros(len(terms))

        for i in range(len(terms)):
            
            tNum = terms.iloc[i, 3]
            dF_ind = neuron.df.index[neuron.df['trial_num'] == tNum]
            ind = np.zeros(len(neuron.df), dtype=bool)
            ind[neuron.df.index[dF_ind[0]]] = 1
            
            t = neuron.get_mean_t((0.1, 0.5))
            cue_fr[i] = np.nanmean(neuron.fr[np.ix_(ind, t)], 1)

            t = neuron.get_mean_t((0.5, 2.0))
            trace_fr[i] = np.nanmean(neuron.fr[np.ix_(ind, t)], 1)

            
        terms['rew_X'] = terms['CUE_TYPE'] * terms['CUE_X']
        terms['rew_Y'] = terms['CUE_TYPE'] * terms['CUE_Y']
        terms['X_Y'] = terms['CUE_Y'] * terms['CUE_X']
        terms['X_Y_rew'] = terms['CUE_TYPE'] * terms['CUE_X'] * terms['CUE_Y']

        terms['cue_fr'] = cue_fr
        terms['trace_fr'] = trace_fr

        frates = terms[['cue_fr', 'trace_fr']]
        frates = frates.stack().reset_index(level=[0,1], drop=True)


        temp = terms.copy()
        terms = pd.concat([terms, temp])

        cue_int = np.empty((len(terms),))
        cue_int[::2] = 1
        cue_int[1::2] = 0

        tr_int = np.empty((len(terms),))
        tr_int[::2] = 0
        tr_int[1::2] = 1

        terms = terms.sort_values(by=['trial_num'])

        terms['cue_int'] = cue_int
        terms['tr_int'] = tr_int

# Calculate the interaction terms

        terms['rew_cue'] = terms['CUE_TYPE'] * terms['cue_int']
        terms['rew_tr'] = terms['CUE_TYPE'] * terms['tr_int']

        terms['cue_X_Y_rew'] = terms['CUE_TYPE'] * terms['CUE_X'] * terms['CUE_Y'] * terms['cue_int']
        terms['tr_X_Y_rew'] = terms['CUE_TYPE'] * terms['CUE_X'] * terms['CUE_Y'] * terms['tr_int']

        terms['FRATE'] = frates

        terms = terms.dropna(how='any')

        PARAMS = ['CUE_TYPE', 'CUE_X', 'CUE_Y', 'cue_int', 'tr_int', 'rew_X', 'rew_Y',
                      'X_Y', 'X_Y_rew', 'rew_cue',
                      'rew_tr', 'cue_X_Y_rew', 'tr_X_Y_rew', 'FRATE']

        sample_xyrew = terms[PARAMS]

        import statsmodels.formula.api as smf

        data = sample_xyrew
        response = 'FRATE'

        """
        Linear model designed by forward selection
        Parameters:
        -----------
        data : pandas DataFrame with all possible predictors and response

        response: string, name of response column in data

        Returns:
        --------
        model: an "optimal" fitted statsmodels linear model
               with an intercept
               selected by forward selection
               evaluated by adjusted R-squared
        """
        remaining = set(data.columns)
        remaining.remove(response)
        selected = []
        current_score, best_new_score = 0.0, 0.0
        while remaining and current_score == best_new_score:
            scores_with_candidates = []
            for candidate in remaining:
                formula = "{} ~ {} + 1".format(response,
                        ' + '.join(selected + [candidate]))
                score = smf.ols(formula, data).fit().rsquared_adj
                scores_with_candidates.append((score, candidate))
            scores_with_candidates.sort()
            best_new_score, best_candidate = scores_with_candidates.pop()
            if current_score < best_new_score:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score
        formula = "{} ~ {} + 1".format(response,
                ' + '.join(selected))
        model = smf.ols(formula, data).fit()

        f.write("Neuron: "+ expName + ", " + nname[0]+ "\n")
        f.write(str(model.summary()) + "\n")


def global_GLM():

    import math
    import statsmodels.api as sm
    from scipy import stats

    io = LoadData()
    os.chdir("/Volumes/AUNOY_SALZ")
    filepath = "/Volumes/AUNOY_SALZ/figs/"
    f = open("GLM_polar_const_results.txt", "w")
    parameters = np.array(np.ones((900, 14)))
    parameters = parameters.astype(float)
    iteration = 0

    for file in glob.glob("*map_*elec*.pickle"):
        filepart = file.split("_")
        expName = filepart[0] + "_" + filepart[1] + "_" +filepart[2]
        nname = filepart[3].split(".")
        print(expName + " " + nname[0])
        neuron = io.load_neuron(expName, nname[0])
        if(expName == 'sn_map_051215' and nname[0]== 'elec7b'):
            continue
        if(expName == 'tn_map_120514' and nname[0] == 'elec19a'):
            continue
        if(expName == 'tn_map_120814' and nname[0] == 'elec10a'):
            continue
        if(expName == 'tn_map_121514' and nname[0] == 'elec14a'):
            continue
        if(expName == 'sn_map_051515' and nname[0] == 'elec16a'):
            continue
        if(expName == 'sn_map_051515' and nname[0] == 'elec16a'):
            continue
        if(expName == 'sn_map_051915' and nname[0] == 'elec29b'):
            continue
        if(expName == 'tn_map_120814' and nname[0] == 'elec6c'):
            continue
       
        neuron.get_xy()
        neuron.get_xy_grid()
        neuron.get_xy_plot_grid()

        x = neuron.x
        y = neuron.y
        neuron.smooth_firing_rates()

        '''
        I need to create a dataframe that only has the r, theta, cue, r-cue,
        theta-cue, r-theta-cue, and fr data.

        '''

        only_good = neuron.df.loc[neuron.df['hit'] == True]

        term = ['CUE_TYPE', 'CUE_X', 'CUE_Y', 'trial_num']
        terms = only_good.loc[:, term]

        terms['r^2'] = terms['CUE_X']**2 + terms['CUE_Y']**2
        terms['theta'] = np.arctan(terms['CUE_Y'] / terms['CUE_X'])

        terms['theta'].fillna((math.pi/2), inplace=True)


        neuron.fr_smooth

        '''

        I want to calculate the mean fr for the cue, trace, and target
        interval. To do this I need to get the average for one trial.
        In terms, I have a list of good trials that I can use.
        As I iterate through, these will have an x condition, y condition
        and a firing rate. What I want to do is access the fr_smooth
        array and get the average firing rate across.  So I need to iterate
        through the good conditions, extract the trial number and cross
        reference to the original df.

        Will wait to look at epoch later.
        '''


        cue_fr = np.zeros(len(terms))
        trace_fr = np.zeros(len(terms))

        for i in range(len(terms)):

            tNum = terms.iloc[i, 3]
            dF_ind = neuron.df.index[neuron.df['trial_num'] == tNum]
            ind = np.zeros(len(neuron.df), dtype=bool)
            ind[neuron.df.index[dF_ind[0]]] = 1

            t = neuron.get_mean_t((0.1, 0.5))
            cue_fr[i] = np.nanmean(neuron.fr[np.ix_(ind, t)], 1)

            t = neuron.get_mean_t((0.5, 2.0))
            trace_fr[i] = np.nanmean(neuron.fr[np.ix_(ind, t)], 1)


        #terms['rew*X'] = terms['CUE_TYPE'] * terms['CUE_X']
        #terms['rew*Y'] = terms['CUE_TYPE'] * terms['CUE_Y']
        #terms['X*Y'] = terms['CUE_Y'] * terms['CUE_X']
        #terms['X*Y*rew'] = terms['CUE_TYPE'] * terms['CUE_X'] * terms['CUE_Y']
        
        terms['rew*R2'] = terms['CUE_TYPE'] * terms['r^2']
        terms['rew*theta'] = terms['CUE_TYPE'] * terms['theta']
        terms['R2*theta'] = terms['r^2'] * terms['theta']
        terms['R2*theta*rew'] = terms['CUE_TYPE'] * terms['r^2'] * terms['theta']
         
        terms['cue_fr'] = cue_fr
        terms['trace_fr'] = trace_fr

        frates = terms[['cue_fr', 'trace_fr']]
        frates = frates.stack().reset_index(level=[0,1], drop=True)


        temp = terms.copy()
        terms = pd.concat([terms, temp])

        cue_int = np.empty((len(terms),))
        cue_int[::2] = 1
        cue_int[1::2] = 0

        tr_int = np.empty((len(terms),))
        tr_int[::2] = 0
        tr_int[1::2] = 1


        terms = terms.sort_values(by=['trial_num'])

        terms['cue_int'] = cue_int
        terms['tr_int'] = tr_int

        # Calculate the interaction terms

        #terms['rew*cue'] = terms['CUE_TYPE'] * terms['cue_int']
        #terms['rew*tr'] = terms['CUE_TYPE'] * terms['tr_int']

        #terms['cue*X*Y*rew'] = terms['CUE_TYPE'] * terms['CUE_X'] * terms['CUE_Y'] * terms['cue_int']
        #terms['tr*X*Y*rew'] = terms['CUE_TYPE'] * terms['CUE_X'] * terms['CUE_Y'] * terms['tr_int']
        
        terms['rew*cue'] = terms['CUE_TYPE'] * terms['cue_int']
        terms['rew*tr'] = terms['CUE_TYPE'] * terms['tr_int']

        terms['cue*R2*theta*rew'] = terms['CUE_TYPE'] * terms['r^2'] * terms['theta'] * terms['cue_int']
        terms['tr*R2*theta*rew'] = terms['CUE_TYPE'] * terms['r^2'] * terms['theta'] * terms['tr_int']
         
        # Calculate the GLM Fit

        depVar = frates

        PARAMS = ['CUE_TYPE', 'r^2', 'theta', 'cue_int', 'tr_int', 'rew*R2', 'rew*theta',
                  'R2*theta', 'R2*theta*rew', 'rew*cue',
                  'rew*tr', 'cue*R2*theta*rew', 'tr*R2*theta*rew']

        sample_xyrew = terms[PARAMS]
        sample_xyrew = sm.add_constant(sample_xyrew)
        glm_xyrew = sm.GLM(depVar, sample_xyrew.values, family=sm.families.Poisson())
        res = glm_xyrew.fit()
        
        f.write("Neuron: "+ expName + ", " + nname[0]+ "\n")
        f.write(str(res.summary()) + "\n")
        #for j in range(len(res.params)):
            #parameters[iteration,j] = res.params[j]
        parameters[iteration] = res.params
        iteration = iteration+1
    
    print(parameters)
    return parameters



def demo():
    io = LoadData()
    os.chdir("/Volumes/AUNOY_SALZ")
    x = []
    y = []
    filepath = "/Volumes/AUNOY_SALZ/figs/"

    for file in glob.glob("*_map_*elec*.pickle"):
        filepart = file.split("_")
        expName = filepart[0] + "_" + filepart[1] + "_" +filepart[2]
        nname = filepart[3].split(".")
        neuron = io.load_neuron(expName, nname[0])
        neuron.get_xy()
        neuron.get_xy_grid()
        neuron.get_xy_plot_grid()

        rewardedness = np.array(np.zeros((5,5)))

        x = neuron.x
        y = neuron.y
        neuron.smooth_firing_rates()
        neuron.get_frmean_by_loc((0.1, 0.5))


        for ix in range(len(neuron.x)):
            for iy in range(len(neuron.y)):
                rew = neuron.frmean_space[1, ix, iy]
                nrew = neuron.frmean_space[0, ix, iy]
                rewardedness[len(neuron.y)-1-iy, ix] = (rew - nrew) / (rew+nrew)

        fig, ax = plt.subplots()
        im, cbar = heatmap(rewardedness, np.flip(y,0), x,  ax=ax, cmap='BrBG', cbarlabel="Reward Preference", vmin=-1, vmax=1)
        texts = annotate_heatmap(im, valfmt="{x:.3g}")


        #ax.set_xticks(np.arange(len(neuron.x)))
        #ax.set_yticks(np.arange(len(neuron.y))) 
        #ax.set_xticklabels(neuron.x)
        #ax.set_yticklabels(neuron.y)
        #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor") 
        #
        #
        #for i in range(len(neuron.x)):
        #    for j in range(len(neuron.y)):
        #        text = ax.text(j, i, round(fr_means[i, j], 2), ha="center", va="center", color="w")
        #
        fig.tight_layout()
        pdf = filepath+expName + nname[0] + "_cue_reward_response_heatmap.pdf"
        plt.savefig(pdf, dpi=150)
        plt.close()


def demo_TRANDCUEINT():
    io = LoadData()
    fr_norm_rewOff= np.array(np.zeros((5,5))) 
    fr_norm_rewOn = np.array(np.zeros((5,5)))
    fr_norm_TRrewOff = np.array(np.zeros((5,5))) 
    fr_norm_TRrewOn = np.array(np.zeros((5,5)))  
    os.chdir("/Volumes/AUNOY_SALZ")
    x = []
    y = []
    filepath = "/Volumes/AUNOY_SALZ/figs/"

    for file in glob.glob("sn_map_*elec*.pickle"):
        filepart = file.split("_")
        expName = filepart[0] + "_" + filepart[1] + "_" +filepart[2]
        nname = filepart[3].split(".")
        neuron = io.load_neuron(expName, nname[0])
        neuron.get_xy()
        neuron.get_xy_grid()
        neuron.get_xy_plot_grid()
        x = neuron.x
        y = neuron.y
        neuron.smooth_firing_rates()
        neuron.get_frmean_by_loc((0.1, 0.5))
        #minn, maxx = neuron.define_hot_spot()
        #neuron.plot_hot_spot(min_pos=minn, max_pos=maxx)

        for irew in range(2):
            for ix in range(len(neuron.x)):
                for iy in range(len(neuron.y)):
                    smin = np.nanmin(neuron.frmean_space[irew, :, :])
                    smax = np.nanmax(neuron.frmean_space[irew, :, :])
                    if(irew):
                        fr_norm_rewOn[len(neuron.y)-1-iy, ix] = neuron.frmean_space[irew, ix, iy] #- smin) / (smax - smin) 
                    else:
                        fr_norm_rewOff[len(neuron.y)-1-iy, ix] = neuron.frmean_space[irew, ix, iy] #- smin) / (smax - smin)
                    #print("Real X, Y = " + str(neuron.x[ix]) + " " + str(neuron.y[iy]))
                    #print("Row, col = " + str((4-iy)) + " " + str(ix))
                    #print("Val = " + str(fr_norm[4-iy, ix]))
           
        neuron.smooth_firing_rates()
        neuron.get_frmean_by_loc((0.5, 2));

        for irew in range(2):
            for ix in range(len(neuron.x)):
                for iy in range(len(neuron.y)):
                    smin = np.nanmin(neuron.frmean_space[irew, :, :])
                    smax = np.nanmax(neuron.frmean_space[irew, :, :])
                    if(irew):
                        fr_norm_TRrewOn[len(neuron.y)-1-iy, ix] = neuron.frmean_space[irew, ix, iy] #- smin) / (smax - smin) 
                    else:
                        fr_norm_TRrewOff[len(neuron.y)-1-iy, ix] = neuron.frmean_space[irew, ix, iy] #- smin) / (smax - smin)


        #fr_plot = np.array([ x/(len(glob.glob("sn*elec17U.pickle"))) for x in fr_means])
        #fr_plot = np.array(fr_means) 
        #print(fr_norm)

        fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8,6))
        #plt.subplot(2,1,1)
        #plt.hist(np.ravel(fr_norm), alpha = 0.5, label = 'norm')
        #plt.subplot(2,1,2)
        #plt.hist(np.ravel(fr_means), alpha = 0.5, label = 'raw')
        im, cbar = heatmap(fr_norm_rewOff, np.flip(y,0), x,  ax=ax2, cmap='Reds', cbarlabel="Cue Mean Firing Rate(sp/s)")
        texts = annotate_heatmap(im, valfmt="{x:.3g}")
        
        im, cbar = heatmap(fr_norm_rewOn, np.flip(y,0), x,  ax=ax, cmap='Blues', cbarlabel="Cue Mean Firing Rate(sp/s)")
        texts = annotate_heatmap(im, valfmt="{x:.3g}") 

        im, cbar = heatmap(fr_norm_TRrewOff, np.flip(y,0), x,  ax=ax4, cmap='Reds', cbarlabel="Trace Mean Firing Rate(sp/s)", label = 'No Rew')
        texts = annotate_heatmap(im, valfmt="{x:.3g}")
        
        im, cbar = heatmap(fr_norm_TRrewOn, np.flip(y,0), x,  ax=ax3, cmap='Blues', cbarlabel="Trace Mean Firing Rate(sp/s)", label = 'Rew')
        texts = annotate_heatmap(im, valfmt="{x:.3g}") 
        

        #ax.set_xticks(np.arange(len(neuron.x)))
        #ax.set_yticks(np.arange(len(neuron.y))) 
        #ax.set_xticklabels(neuron.x)
        #ax.set_yticklabels(neuron.y)
        #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor") 
        #
        #
        #for i in range(len(neuron.x)):
        #    for j in range(len(neuron.y)):
        #        text = ax.text(j, i, round(fr_means[i, j], 2), ha="center", va="center", color="w")
        #
        red_patch = mpatches.Patch(color='red', label='No Reward')
        blue_patch= mpatches.Patch(color='blue', label='Reward') 
        fig.legend(handles=[red_patch, blue_patch], loc='upper right')
        fig.tight_layout()
        pdf = filepath+expName + nname[0] + "_CUE_TR_mean_fr_heatmap.pdf"
        plt.savefig(pdf, dpi=150)
        plt.close()


def demoTime():
    #intTime = []
    #cueOn = []
    #fixOn = []
    #fixAch = []
    #acInt = []
    #fcInt = []
    #noFix = []
    #traceInt = []
    #targInt = []
    fixOut = []
    saccade = []
    io = LoadData()
    os.chdir("/Volumes/AUNOY_SALZ")
    for file in glob.glob("*elec*.pickle"):
        filepart = file.split("_")
        expName = filepart[0] + "_" + filepart[1] + "_" +filepart[2]
        nname = filepart[3].split(".")
        neuron = io.load_neuron(expName, nname[0])
        for ind, spks in neuron.df['spkdata'].iteritems():
            #cueInt = (neuron.df['t_CUE_OFF'].loc[ind] - neuron.df['t_CUE_ON'].loc[ind])
            #cueSt = neuron.df['t_CUE_ON'].loc[ind]
            #fOn = neuron.df['t_FP_ON'].loc[ind]
            #fA = neuron.df['t_FIX_ACH'].loc[ind]
            #tInt = (neuron.df['t_targ_on'].loc[ind] - neuron.df['t_cue_off'].loc[ind]) 
            #tgInt = (neuron.df['t_TARG_OFF'].loc[ind] - neuron.df['t_TARG_ON'].loc[ind])  
            fO = (neuron.df['t_FIX_OUT'].loc[ind] - neuron.df['t_TARG_OFF'].loc[ind]) 
            sac = (neuron.df['t_TARG_ACH'].loc[ind] - neuron.df['t_FIX_OUT'].loc[ind])  
            #ac = cueSt - fA
            #fc = cueSt - fOn
            #nF = neuron.df['t_NO_FIX'].loc[ind]
            if (fO > 0):
                fixOut.append(fO)
            if (sac > 0):
                saccade.append(sac)
            #if (tInt > 0):
            #    traceInt.append(tInt)
            #if (tgInt >0):
            #    targInt.append(tgInt)
            #if (nF > 0):
            #    noFix.append(nF);
            #if (ac >0):
            #    acInt.append(ac)
           # if (fc > 0):
           #     fcInt.append(fc)
            #if (cueInt >0):
                #print (cueInt)
            #    intTime.append(cueInt)
            #    cueOn.append(cueSt)
            #if (fOn >= 0):
            #    fixOn.append(fOn)
            #if (fA >= 0):
            #    fixAch.append(fA)
    #minAC = np.nanmin(acInt)
    #minFC = np.nanmin(fcInt)
    #minFix = np.nanmin(fixOn)
    #minInt = np.nanmin(traceInt)
    #minTr = np.nanmin(traceInt)
    #tintTime = [x - minTr for x in traceInt]
    #print(minTr)
    #mincounts = [x - minAC for x in acInt]
    #avlam = np.nanmean(tintTime) 
    #cueOn = [x - minIntTime for x in cueOn]
    #print(avlam)
    #print(np.nanmean(traceInt))
    #print(minAC)
    #print(minFC)
    #print(minFix)
    #bins = np.linspace(0, 2, 100)
    #x = range(1, len(mincounts))
    #popt, pcov = curve_fit(expo_fun, x, mincounts

    
    plt.figure()
    #stats.probplot(traceInt, dist='expon', fit = True, plot = plt)
    #plt.hist(cueOn, bins, alpha = 0.5, label= 'CueOnset')
    #plt.hist(traceInt, alpha = 0.5, label = 'trace int')
    plt.hist(fixOut, alpha = 0.5, label = 'target to fix break') 
    plt.hist(saccade, alpha = 0.5, label = 'fix break to sacc') 
    #plt.hist(targInt, alpha = 0.5, label = 'target int')
    #plt.hist(fixOn, bins, alpha = 0.5, label= 'FixOnset')
    #plt.hist(mincounts, bins, alpha = 0.5, label = 'Dist of add')
    #plt.hist(fixAch, bins, alpha = 0.5, label= 'FixAch') 
    #plt.hist(mincounts, bins, alpha = 0.5, label = 'Ach-Cue interval')
    #plt.hist(noFix, alpha = 0.5, label = 'No Fix achieved') 
    #plt.hist(mincounts, alpha = 0.5, label = 'Distribution of variable term cue onset')  
    #plt.hist(fcInt, bins, alpha = 0.5, label = 'Fon-Cue interval') 
    plt.legend(loc='upper right')
    plt.ylabel('Counts')
    plt.xlabel('Time (s)')
    plt.show()

def get_file_info():
    # initialize Experiment, and load information
    finfo = sp.io.loadmat(
        os.path.join(os.sep, 'Volumes/AUNOY_SALZ/src/map_cell_list.mat'), 
        squeeze_me=True
    )
    # changed to zero based inds
    finfo['cell_ind'] -= 1
    finfo['file_ind'] -= 1
    return finfo

def create_all(overwrite=True):
    ''' Create all Sessions & Neuron objects '''
    finfo = get_file_info()
    directory = os.path.join(os.sep, 'Volumes/AUNOY_SALZ/src/')
    exp = Experiment(overwrite)
    for iFile, file in enumerate(finfo['filenames']):
        if not file in exp.files or overwrite:
            f = sp.io.loadmat(directory + file + '.nex.mat', squeeze_me=True)
            print('Loaded file', file)
            cells = finfo['cell_name'][finfo['file_ind'] == iFile]
            exp.add_session(f, file, cells)
    exp.save_experiment()

def overwrite_sessions():
    ''' Create all Sessions & Neuron objects '''
    finfo = get_file_info()
    directory = os.path.join(os.sep, 'Volumes/AUNOY_SALZ/src/')
    for iFile, fname in enumerate(finfo['filenames']):
        f = sp.io.loadmat(directory + fname + '.nex.mat', squeeze_me=True)
        print('Loaded file', fname)
        session = Session(f, fname, [])
        session.save_session()

def load_neurons_and_plot(start_ind=0):
    ''' load all Neuron objects and plot them '''
    from tqdm import tqdm
    io = LoadData()
    fig_dir = os.path.join(os.sep, 'Volumes/AUNOY_SALZ/figs/')
    for i, file in tqdm(enumerate(io.experiment.files)):
        if i >= start_ind:
            for cell in io.experiment.files[file]:

                # load neuron object
                print('Loading neuron', file, cell)
                neuron = io.load_neuron(file, cell)
                with PdfPages(fig_dir + neuron.filename + '_' + neuron.name +
                              '.pdf') as pdf:

                    # X/Y information for stimuli in mapping experiment
                    neuron.get_xy()
                    neuron.get_xy_grid()
                    neuron.get_xy_plot_grid()

                    # Computer firing rates for each location in experiment
                    neuron.smooth_firing_rates()
                    neuron.get_frmean_by_loc((.1,.5))
                    neuron.get_fr_by_loc()

                    #neuron.psth_map(pdf)
                    #neuron.psth_rew(pdf)

                    min_pos, max_pos = neuron.define_hot_spot()
                    neuron.plot_hot_spot(min_pos, max_pos)

                    # Use firing rates to determine initial guess paramaters for 2d
                    # gaussian fit
                    #neuron.get_initial_params()

                    # Fit guassian function with 'basinhopping' algorithm in attempt
                    # find a global minimum in this paramater space
                    #neuron.fit_gaussian(niter=1)

                    # Get high-res guassian for plotting based on the fitted parameters
                    #g_nr = neuron.get_gauss(neuron.x_plot_grid, neuron.y_plot_grid, 0,
                    #                        neuron.betas)
                    #g_rw = neuron.get_gauss(neuron.x_plot_grid, neuron.y_plot_grid, 1,
                    #                        neuron.betas)
                    #neuron.plot_gaussian([g_nr, g_rw], neuron.frmean_space, pdf)

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


if __name__ == '__main__':

    #create_all(overwrite=True)

    #load_neurons_and_plot(start_ind=0)
    #create_all(overwrite=True)
    #overwrite_sessions()
    demo()
