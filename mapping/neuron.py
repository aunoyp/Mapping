# -*- coding: utf-8 -*-

import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize
import scipy.signal


class Neuron(object):
    '''Neuron object: Contrains all data relevant to analyzing neural spiking

    -self.df: Pandas DataFrame containing all trial information needed for
        analyzing task-dependent modulation of neural firing rates, as well
        timestamps of neural spikes
    -self.fr: Numpy Array containin firing rates for all trials in the shape
        (# of trials x number of time bins)
    -self.tFrame: Time intervals within which to compute firing rates relative
        to 't_CUE_ON'
    -self.tInt: size of firing rate time bins
    -self.tShift: time shift between successive time bins

    -ver3: same std for reward and no reward now
    '''

    def __init__(self, f, fname, cellname):

        self.directory = os.path.join(os.sep, 'Volumes/AUNOY_SALZ/')

        # cell info
        self.filename = fname
        self.name = cellname
        self.df = None

        # parameters for firing rates, using time in milliseconds (as integers)
        # to avoid floating error
        self.tFrame = [-500, 1500]
        self.tInt = 10
        self.tShift = self.tInt
        if self.tInt < self.tShift:
            print('compute_firing_rates not set up for overlapping bins')
            return
        self.tStart = np.linspace(self.tFrame[0], self.tFrame[1] - self.tInt,
                                  (self.tFrame[1] - self.tFrame[0]) / self.tShift)

        # Chris is saying that the t.Start time is an array that has evenly
        # spaced numbers over the time interval of {tFrame[0], tFrame[1] - 
        # self.tInt} with (tFrame[1]-tFrame[0]) / t.Shift elements. 
        # tFrame[0]= -500 seconds
        # tFrame[1] = 1500
        # tShift = self.tInt
        # self.tInt = 10
        # This means that the linspace for all neurons is from -500 to 1490
        # And the number of elements is 2000/10 = 200 elements.

        self.tEnd = np.linspace(self.tFrame[0] + self.tInt, self.tFrame[1],
                                (self.tFrame[1] - self.tFrame[0]) / self.tShift)
        # Chris is defining here the tEnd to be an evenly spaced interval
        # of -490 to 1500 and the number of elements are also 200


        # covert to seconds (floats)
        self.tFrame = [t / 1e3 for t in self.tFrame]
        self.tInt /= 1e3
        self.tShift /= 1e3
        self.tStart /= 1e3
        self.tEnd /= 1e3
        if len(self.tStart) != len(self.tEnd):
            print('bad time window')
            return

        # smoothed firing rates
        self.fr_smooth = None
        self.tStart_smooth = None
        self.tEnd_smooth = None

        # location firing rates
        self.fr_space = None
        self.frmean_space = None

        # fit parameters for 2d gaussian
        self.param_labels = ['ux', 'uy', 'stdx', 'stdy',
                             'minfr_nr', 'maxfr_nr', 'minfr_rw', 'maxfr_rw']
        self.lb = [-40, -40, 0, 0,
                   0, 0, 0, 0]
        self.ub = [40, 40, 100, 100,
                   200, 200, 200, 200]
        self.betas = None

        # plotting params for 2d gaussians
        self.xy_max = 20
        self.plot_density = 10 #points per degree
        self.x = None
        self.y = None
        self.x_plot = None
        self.y_plot = None
        self.x_plot_grid = None
        self.y_plot_grid = None

        # need something to say whether neuron is MUA or SUA
        if self.name[-2:] == 'wf':
            print('NEURON HAS _WF SUFFIX')
        self.SUA = self.name[-1] != 'U'

        # which side of the screen is contralateral to the recording site?
        # left: -1, right: 1

        tmp = f['dat'][()]['data']['CUE_X'][f['dat'][()]['data']['CUE_CONTRA'] == 1]
        tmp = np.unique(np.array(np.sign(tmp), dtype=int))
        if len(tmp)==1:
            self.contra = tmp[0]
        else:
            print('cant determine which side is conta')

        # get the the data
        self.process(f, fname, cellname)

    def __str__(self):
        string = ''
        string += '\n' + 'self.filename: ' + str(self.filename)
        string += '\n' + 'self.name: ' + str(self.name)
        string += '\n' + 'self.ntrials: ' + str(self.ntrials)
        string += '\n' + 'self.depth: ' + str(self.depth)
        string += '\n' + 'self.AP: ' + str(self.AP)
        string += '\n' + 'self.ML: ' + str(self.ML)
        return string

    def process(self, f, fname, cellname):
        '''Extract non trial-by-trial information about the neuron'''
        # need the index of the cell in that file
        cell_ind = np.nonzero(f['dat'][()]['spkNames'] == cellname)[0]
        if len(cell_ind) == 1:
            cell_ind = cell_ind[0]
        else:
            print('wrong # of cells')
            return

        # general cell info
        self.ntrials = len(f['dat'][()]['data'])
        self.NPointsWave = f['dat'][()]['NPointsWave'][cell_ind]
        self.WFrequency = f['dat'][()]['WFrequency'][cell_ind]
        self.WLength = f['dat'][()]['WLength'][cell_ind]
        self.AP = f['dat'][()]['session_info'][()]['AP']
        self.ML = f['dat'][()]['session_info'][()]['ML']
        base_depth = f['dat'][()]['session_info'][()]['brain_depth'] + \
                     f['dat'][()]['session_info'][()]['guide_depth']

        # spike depth is one/trial - reduce to one/session
        depth = set([f['dat'][()]['data'][()]['spk_depth'][i][cell_ind]
                for i in range(self.ntrials)])
        if len(depth) == 1:
            self.depth = base_depth + depth.pop()
        else:
            print('multiple depths for one neuron')
            return

        ### trial data desired
        labels = self.get_data_labels()
        self.extract_trial_info(f, cell_ind, labels)
        self.compute_firing_rates()

        # plot definitions
        self.rew_colors = ('r','b')

    def get_data_labels(self):
        '''Data fields that we want to keep from the loaded .mat file.
        Specify desired data type for each field as the default is 'object'
        '''
        return [('spkdata', 'float'),
                ('t_FP_ON', 'float'), ('t_FIX_ACH', 'float'),
                ('t_CUE_ON', 'float'), ('t_CUE_OFF', 'float'),
                ('t_TARG_ON', 'float'), ('t_TARG_OFF', 'float'),
                ('t_FIX_OUT', 'float'), ('t_TARG_ACH', 'float'),
                ('t_SUCCESS', 'float'), ('t_REWARD_ON', 'float'),
                ('t_REWARD_OFF', 'float'), ('t_OUTCOME_OFF', 'float'),
                ('t_TRIAL_END', 'float'), ('t_NO_FIX', 'float'),
                ('t_BREAK_FIX', 'float'), ('t_BREAK_CUE', 'float'),
                ('t_BREAK_TARG', 'float'), ('t_NO_SACCADE', 'float'),
                ('t_MISS_TARG', 'float'), ('t_NO_HOLD_TARG', 'float'),
                ('t_FAIL', 'float'), ('TARG_WIDTH', 'float'),
                ('TARG_CONTRAST', 'float'), ('TARG_FREQ', 'float'),
                ('CUE_X', 'float'), ('CUE_Y', 'float'),
                ('CUE_WIDTH', 'float'), ('CUE_TYPE', 'float'),
                ('REPEAT', 'bool')]

    def extract_trial_info(self, f, cell_ind, labels):
        '''Process trial information and spikes from the .mat structure. First
        pass this information into a python dict and then transform into a
        pandas datafraem.

        Adds additional trial information fields for easier indexing in future
        data analysis

        Only keeps trials where the neuron was recorded clean ('good_trial')
        but keep an index ('trial_num') of the trial number within that
        session. This is necessary for pulling corresponding analog data from
        corresponding 'Session' object
        '''
        # trials where the recording of this neuron is valid
        # ONLY these trials will be saved
        good_trial = np.where(
                    [f['dat'][()]['data'][()]['good_trial'][i][cell_ind]==True
                    for i in range(self.ntrials)])[0]

        # DICT with all trial information
        # all information is the same (one per trial) EXCEPT for 'spkdata'
        # which can be any number of elements per trial
        # (these are all CAST as OBJECTS - need to specify int or float)
        data = {}
        for label in labels:
            if label[0] == 'spkdata':
                data[label[0]] = []
                for iTrial in good_trial:
                    # spks will be a LIST of ARRAYS
                    spks = f['dat'][()]['data'][label[0]][iTrial][cell_ind]
                    # if there is one element, scipy interprets as a float
                    # else (len=0 or >1): np.array
                    if type(spks) == float:
                        data[label[0]].append(np.array([spks], dtype=label[1]))
                    else:
                        data[label[0]].append(np.array(spks, dtype=label[1]))
            elif label[0] == 't_REWARD_ON':
                # 't_REWARD_ON' may include mutliple values - include the 1st
                data[label[0]] = np.full_like(good_trial, np.nan, dtype=label[1])
                for i, iTrial in enumerate(good_trial):
                    times = f['dat'][()]['data'][label[0]][iTrial]
                    if type(times) == float:
                        data[label[0]][i] = times
                    else:
                        data[label[0]][i] = times[0]
            else:
                data[label[0]] = np.array(
                    f['dat'][()]['data'][label[0]][good_trial], dtype=label[1])


        # add additional fields
        data['trial_num'] = good_trial
        data['rew'] = data['CUE_TYPE'] % 2
        data['cue_set'] = data['CUE_TYPE']
        data['cue_set'][np.logical_or(data['cue_set'] == 1, data['cue_set'] == 2)] = 0
        data['cue_set'][np.logical_or(data['cue_set'] == 3, data['cue_set'] == 4)] = 1
        data['hit'] = np.logical_not(np.isnan(data['t_SUCCESS']))
        x0 = np.logical_not(np.isnan(data['t_NO_SACCADE']))
        x1 = np.logical_not(np.isnan(data['t_MISS_TARG']))
        x2 = np.logical_not(np.isnan(data['t_NO_HOLD_TARG']))
        data['miss'] = np.logical_or(x0, x1)
        data['miss'] = np.logical_or(data['miss'], x2)

        # tranfsorm DICT into a DATAFRAME
        self.df = pd.DataFrame(data)

    def compute_firing_rates(self):
        ''' Given spike timestamps in the dataframe, create a numpy array
        with firing rate for every trial

        Exclude all data after 'end' events (set to np.nan)

        10-fold decrese in run time compared to old method. Does not work
        with overlapping time windows, but can do this with subsequent call
        to 'smooth_firing_rates'
        '''
        # Define self.fr -> array of size of trials by 200 bins
        self.fr = np.full((self.df.shape[0], len(self.tStart)), np.nan)

        # so with iterator indices, and spikes
        for ind, spks in self.df['spkdata'].iteritems():
            # cueOn is equal to the time when the cue is on
            cueOn = self.df['t_CUE_ON'].loc[ind]
            if not np.isnan(cueOn):    # if the cue does not turn on
                start_ind = 0    # startind is equal to zero
                end_time = np.nanmin([self.df['t_TARG_ON'].loc[ind] - cueOn,
                                self.df['t_BREAK_TARG'].loc[ind] - cueOn,
                                self.df['t_BREAK_CUE'].loc[ind] - cueOn,
                                self.df['t_TRIAL_END'].loc[ind] - cueOn])
                # end_time is equal to the smallest value of TargOn
                # Break Targ, Break Cue, or Trial End.
                # Whatever comes first will be the end time
                for t in range(len(self.tStart)):    # iterate 200 times
                    # if time frame is after an 'end' event, fr = nan
                    if end_time < self.tEnd[t]:
                        break
                    # if at end of spike list, fr = 0
                    if start_ind == len(spks):
                        self.fr[ind, t] = 0
                        continue
                    # counts spikes in this time window
                    curr_ind = start_ind
                    nspks = 0
                    while (curr_ind < len(spks) and
                           spks[curr_ind] - cueOn < self.tEnd[t]):
                        if spks[curr_ind] - cueOn >= self.tStart[t]:
                            nspks +=1
                        curr_ind +=1
                    # compue firing rate
                    self.fr[ind, t] = nspks / self.tInt # which is nspks / 10 sec int
                    # set the start point for searching for spikes to be first
                    # spike after the current time window
                    start_ind = curr_ind

    ### ANYTHING ABOVE HERE REQUIRES RE-GENERATING THE DATA FILES

    def smooth_firing_rates(self, binSize=10, binShift=1):
        ''' Given firing rates in self.fr in non-overlapping bins, compute
        a smoothed version of firing rate in self.fr_smooth

        -binSize: number of bins in self.fr to average over for a bin in
            self.fr_smooth
        -binShift: shift between bins in self.fr for computing smoothed
            firing rates
        e.g. self.tInt = .01, self.tShift = .01, binSize = 10, binShift = 1
        gives new bin sizes of .1 (s) shifted by .01 (s)

        This method works efficiently by transforming the (M, N) array of
        firing rates into a (M*N,) array and then using a FFT convolution via
        SciPy. Output is reshaped into a (M, N') array where N' is the
        resulting number of time bins.

        nan's are treated such that any time bin in self.fr_smooth that
        includes data from a bin=nan in self.fr is also equal to nan. More
        formally, self.fr[i, t]==nan => self.fr_smooth[i, t:t+binSize] == nan
        '''
        M = np.size(self.fr,0)
        N = np.size(self.fr,1)
        if binSize <= 0 or binSize > N:
            return

        # need to get rid of NaNs before passing to 'fftconvolve'
        # keep track of their location, so they can be re-NaNed later
        nan_inds = np.isnan(self.fr)
        # flattten and convolve
        x_flat = np.array(self.fr)
        x_flat[nan_inds] = 0
        x_flat = x_flat.ravel()
        x_conv = sp.signal.fftconvolve(x_flat,
                                       np.ones((binSize,)) / binSize, mode='valid')

        self.fr_smooth = np.full((M, N - (binSize-1)), np.nan)
        i = 0 #indice in flattened array
        row = 0 #row in out put matrix
        while i < len(x_conv):
            self.fr_smooth[row,:] = x_conv[i : i + (N - (binSize - 1))]
            i += N
            row += 1

        shifted = np.array(nan_inds)
        for i in range(1, binSize):
            shifted = np.hstack((shifted[:,1:], np.full((M,1), False, dtype=bool)))
            nan_inds = nan_inds | shifted
        nan_inds = nan_inds[:, :N-(binSize-1)]
        self.fr_smooth[nan_inds] = np.nan

        self.fr_smooth = self.fr_smooth[:, ::binShift]
        self.tStart_smooth = self.smooth_time_points(self.tStart, binSize, binShift)
        self.tEnd_smooth = self.smooth_time_points(self.tEnd, binSize, binShift)

    def smooth_time_points(self, t, binSize, binShift):
        '''After smoothing firing rates, get the corresponding time windows'''
        return np.convolve(t, np.ones((binSize,)) / binSize,
                           mode='valid')[::binShift]

    def get_fr_by_loc(self):
        ''' Computes the firing rate relative to cue onset for each spatial
        location
        '''
        self.fr_space = np.empty((2, len(self.tStart_smooth),
                                  len(self.x), len(self.y)))
        for ix in range(len(self.x)):
            for iy in range(len(self.y)):
                for irew in range(2):
                    inds = self.get_xy_rew_inds(self.x[ix], self.y[iy], irew)
                    self.fr_space[irew, :, ix, iy] = np.nanmean(self.fr_smooth[inds,:], 0)

    def get_frmean_by_loc(self, tMean):
        ''' Computes the mean firing rate for each spatial location '''
        t = self.get_mean_t(tMean)
        self.frmean_space = np.empty((2, len(self.x), len(self.y)))
        for ix in range(len(self.x)):
            for iy in range(len(self.y)):
                for irew in range(2):
                    inds = self.get_xy_rew_inds(self.x[ix], self.y[iy], irew)
                    self.frmean_space[irew, ix, iy] = \
                        np.nanmean(np.mean(self.fr[np.ix_(inds, t)], 1))

    ### AUXILLARY FUNCTIONS
    def get_gauss(self, x, y, is_rew, betas0):
        ''' g = min(4) + max-min(5) *
            exp( - ((x-ux(0))/stdx(2))**2 - ((y-uy(1))/stdy(3))**2)
        '''
        if is_rew:
            k = 2
        else:
            k = 0
        x_part = ((x - betas0[0]) / betas0[2]) ** 2
        y_part = ((y - betas0[1]) / betas0[3]) ** 2
        g = betas0[k+4] + betas0[k+5] * np.exp(-x_part/2 - y_part/2)
        return g

    def get_initial_params(self):

        #initialize
        self.betas = np.full_like(self.param_labels, None, dtype=float)
        # flatten
        x = np.ravel(self.x_grid)
        y = np.ravel(self.y_grid)
        z0 = np.ravel(self.frmean_space[0])
        z1 = np.ravel(self.frmean_space[1])
        z = np.nanmean(np.vstack((z0, z1)), 0)
        #ind of max firing rate across reward conditions
        ind = np.argmax(z)

        self.betas[0] = x[ind]
        self.betas[1] = y[ind]
        self.betas[2] = 20
        self.betas[3] = 20
        self.betas[4] = np.nanmin(z0)
        self.betas[5] = np.nanmax(z0) - np.nanmin(z0)
        self.betas[6] = np.nanmin(z1)
        self.betas[7] = np.nanmax(z1) - np.nanmin(z1)

    def error_func(self, betas0):
        ''' Error function of fitted 2d gaussian: returns sums of squares for
        the current set of parameters. If parameters are out of bounds,
        returns 'np.inf'.
        '''
        if np.any(betas0 < self.lb) or np.any(betas0 > self.ub):
            return np.inf
        x = np.ravel(self.x_grid)
        y = np.ravel(self.y_grid)
        z0 = np.ravel(self.frmean_space[0])
        z1 = np.ravel(self.frmean_space[1])
        return np.nansum(np.r_[z0 - self.get_gauss(x, y, 0, betas0),
                               z1 - self.get_gauss(x, y, 1, betas0)] ** 2)

    def fit_gaussian(self, niter=100, print_betas=True):
        betas0 = copy.copy(self.betas)
        results = sp.optimize.basinhopping(self.error_func, self.betas,
                                           disp=False,
                                           accept_test=self.eval_fit,
                                           niter=niter)
        self.betas = results.x
        # print('\tSTART\tFINISH')
        # for i in range(len(betas0)):
        #     print('%s:\t%.2f\t%.2f' % (self.param_labels[i],
        #                                betas0[i], self.betas[i]))
        return results

    def take_step(self, x0):
        xy_inds = [('ux' in label) or ('uy' in label) for label in
                   self.param_labels]
        std_inds = ['std' in label for label in self.param_labels]
        min_inds = ['minfr' in label for label in self.param_labels]
        max_inds = ['maxfr' in label for label in self.param_labels]
        x = copy.copy(x0)
        x[np.where(xy_inds)] = np.random.uniform(-20, 20, np.sum(xy_inds))
        x[np.where(std_inds)] = np.random.uniform(1, 80, np.sum(std_inds))
        x[np.where(min_inds)] = np.random.uniform(1, 100, np.sum(min_inds))
        x[np.where(max_inds)] = np.random.uniform(1, 100, np.sum(max_inds))
        return x

    def print_fun(self, x, f, accepted):
        if not np.isinf(f) and accepted == 1:
            #g_nr = self.get_gauss(self.x_plot_grid, self.y_plot_grid, 0, x)
            #g_rw = self.get_gauss(self.x_plot_grid, self.y_plot_grid, 1, x)
            #self.plot_gaussian([g_nr, g_rw], self.frmean_space)
            print('%.2f\t' * len(x) %tuple(x))
            print('   f = %1.2f, accepted = %d' %(f, accepted))

    def eval_fit(self, f_new, x_new, f_old, x_old):
        ''' Check that the parameters corresponding to this local mininium are
        within in bounds. This called after each basinhopping iteration.

        -f_new, f_old: error for the current/last set of parameters, as defined
         'error_func' (sums of squares in this case)
        -x_new, x_old: new parameters for this iteration, and old parameters
         for the best fit so far
        '''
        accept = (np.all(x_new >= self.lb) and np.all(x_new <= self.ub) and
                  f_new < f_old)
        return bool(accept)

    def get_xy(self):
        ''' all possible x and y locations of the stimuli'''
        self.x = np.unique(self.df['CUE_X'])
        self.x = self.x[np.isnan(self.x)==False]
        self.y = np.unique(self.df['CUE_Y'])
        self.y = self.y[np.isnan(self.y)==False]

    def get_xy_grid(self):
        ''' Grid coordinates for all  x and y locations of the stimuli'''
        self.x_grid = np.full([len(self.x), len(self.y)], 0, dtype=float)
        self.y_grid = np.full([len(self.x), len(self.y)], 0, dtype=float)
        for xi in range(len(self.x)):
            for yi in range(len(self.y)):
                self.x_grid[xi, yi] = self.x[xi]
                self.y_grid[xi, yi] = self.y[yi]

    def get_xy_plot_grid(self):
        ''' Grid coordinates for plotting 2d Gaussian fit'''
        x_plot = np.linspace(-self.xy_max, self.xy_max,
                             2 * self.plot_density * self.xy_max + 1)
        y_plot = x_plot
        # grid of locations for plotting
        self.x_plot_grid = np.full([len(x_plot), len(y_plot)], 0, dtype=float)
        self.y_plot_grid = np.full([len(x_plot), len(y_plot)], 0, dtype=float)
        for xi in range(len(x_plot)):
            for yi in range(len(y_plot)):
                self.x_plot_grid[xi, yi] = x_plot[xi]
                self.y_plot_grid[xi, yi] = y_plot[yi]

    def get_mean_t(self, tWin):
        ''' Indices in raw firing rates to take for computing the mean firing
        rate in particular time window (tWin)'''
        return np.logical_and(self.tStart - tWin[0] > -.0001,
                              self.tEnd - tWin[1] < 0.0001)

    def get_ylim(self):
        ''' For plotting, find the max and min firing rates across all
        spatial locations and reward conditions
        '''
        ymin = int(np.floor(np.nanmin(self.fr_space)))
        ymax = int(np.ceil(np.nanmax(self.fr_space)))
        return (ymin, ymax)

    def get_xy_rew_inds(self, x, y, irew):
        ''' Returns indices of trials corresponding to a particular spatial
        location (x, y) and reward condition (irew). 'eps' is the allowable
        float point error for testing equality'''
        eps = 0.01
        is_x = np.array(abs(self.df['CUE_X'] - x) < eps)
        is_y = np.array(abs(self.df['CUE_Y'] - y) < eps)
        is_rew = np.array(self.df['rew'] == irew)
        return np.logical_and(np.logical_and(is_x, is_y), is_rew)

    ### PLOT FUNCTIONS
    def psth(self, binSize=10, binShift=1):
        ''' Plot of raw vs. smooth firing rates'''
        if self.fr_smooth == None:
            self.smooth_firing_rates(binSize, binShift)
        t0 = np.mean(np.c_[self.tStart, self.tEnd],1)
        t1 = np.mean(np.c_[self.tStart_smooth, self.tEnd_smooth], 1)
        fr0 = np.nanmean(self.fr, 0)
        fr1 = np.nanmean(self.fr_smooth, 0)
        plt.figure()
        plt.plot(t0, fr0)
        plt.plot(t1, fr1)
        plt.show()
    
    


    def psth_horiz(self, binSize=10, binShift=1, pdf=None):
        ''' Firing rates relative to cue onset for cue predicted reward and
        no reward (irrespective of spatial location)
        '''
        self.get_xy()
        self.get_xy_grid()
        self.get_xy_plot_grid()
        
        self.smooth_firing_rates()
        self.get_frmean_by_loc((0.1, 0.5))
        self.get_fr_by_loc()
        
        t = np.mean(np.c_[self.tStart_smooth, self.tEnd_smooth], 1)

        fig, ax = plt.subplots(nrows=2, ncols=1)
        plt.sca(ax[0])
        plt.plot(t, self.fr_space[1, :, 4, 0], c=self.rew_colors[0])
        plt.plot(t, self.fr_space[1, :, 2, 0], c=self.rew_colors[1])
        plt.ylabel('Firing rate (sp/s)')

        
        plt.sca(ax[1])
        count = 1
        for xi in range(len(self.x)):  
            for i in np.where(self.df['rew']==1)[0]:
                if(self.x[xi] > 0):
                    spks = copy.copy(self.df.ix[i, 'spkdata']-
                        self.df.ix[i, 't_CUE_ON'])
                    spks = spks[(spks >= self.tFrame[0]) & (spks <= self.tFrame[1])]
                    plt.scatter(spks, [count]*len(spks), c=self.rew_colors[1],
                            marker='|', linewidth=0.1)
                    count+=1
                #elif (self.x[ix] < 0):


        #for is_rew in range(2):
        #    for i in np.where(self.df['rew']==is_rew)[0]:
        #        spks = copy.copy(self.df.ix[i, 'spkdata'] -
        #                         self.df.ix[i, 't_CUE_ON'])
        #        spks = spks[(spks >= self.tFrame[0]) & (spks <= self.tFrame[1])]
        #        plt.scatter(spks, [count]*len(spks), c=self.rew_colors[is_rew],
        #                    marker='|', linewidth=0.1)
        #        count += 1
        plt.xlim(self.tFrame)
        plt.ylim((0, count))
        plt.xlabel('Time relative to cue onset (s)')
        plt.ylabel('Trial #')
        if pdf:                  
            pdf.savefig()
            plt.close()
        else:
            plt.show()
     
    def psth_map(self, binSize=10, binShift=1, pdf=None):
        ''' Firing rates relative to cue onset for each reward condition
        (reward or no reward) and spatial location
        '''
        ylim = self.get_ylim()
        t = np.mean(np.c_[self.tStart_smooth, self.tEnd_smooth], 1)
        fig, ax = plt.subplots(nrows=len(self.y), ncols=len(self.x))
        for xi in range(len(self.x)):
            for yi in range(len(self.y)):
                #plot
                plt.sca(ax[len(self.y) - yi - 1, xi])
                plt.plot(t, self.fr_space[0, :, xi, yi],
                         color=self.rew_colors[0], lw=1)
                plt.plot(t, self.fr_space[1, :, xi, yi],
                         color=self.rew_colors[1], lw=1)
                plt.plot((0,0), ylim, linestyle='--', color='0.5')
                #format                       
                plt.title('x=%1.1f, y=%1.1f' % (self.x[xi], self.y[yi]), size=6)
                if yi == 0:
                    plt.xticks(size=4)
                else:
                    plt.xticks([])
                if xi == 0:
                    plt.yticks(size=5)
                else:
                    plt.yticks([])
                plt.xlim(self.tFrame)
                plt.ylim(ylim)
                plt.box()
        title = self.filename + '_' + self.name
        fig.text(0.5, 0.99, title,
                 ha='center', va='center')
        fig.text(0.5, 0.01, 'Time relative to cue onset (s)',
                 ha='center', va='center')
        fig.text(0.01, 0.5, 'Firing rate (sp/s)',
                 ha='center', va='center', rotation='vertical')
        fig.tight_layout()

        if pdf:
            plt.savefig(pdf, dpi=150)
            plt.close()
        else:
            plt.show()

    def psth_rew(self, binSize=10, binShift=1, pdf=None):
        ''' Firing rates relative to cue onset for cue predicted reward and
        no reward (irrespective of spatial location)
        '''
        t = np.mean(np.c_[self.tStart_smooth, self.tEnd_smooth], 1)
        fr0 = np.nanmean(self.fr_smooth[np.array(self.df['rew']==0),:], 0)
        fr1 = np.nanmean(self.fr_smooth[np.array(self.df['rew']==1),:], 0)

        fig, ax = plt.subplots(nrows=2, ncols=1)
        plt.sca(ax[0])
        plt.plot(t, fr0, c=self.rew_colors[0])
        plt.plot(t, fr1, c=self.rew_colors[1])
        plt.ylabel('Firing rate (sp/s)')

        plt.sca(ax[1])
        count = 1
        for is_rew in range(2):
            for i in np.where(self.df['rew']==is_rew)[0]:
                spks = copy.copy(self.df.ix[i, 'spkdata'] -
                                 self.df.ix[i, 't_CUE_ON'])
                spks = spks[(spks >= self.tFrame[0]) & (spks <= self.tFrame[1])]
                plt.scatter(spks, [count]*len(spks), c=self.rew_colors[is_rew],
                            marker='|', linewidth=0.1)
                count += 1
        plt.xlim(self.tFrame)
        plt.ylim((0, count))    
        plt.xlabel('Time relative to cue onset (s)')
        plt.ylabel('Trial #')
        if pdf:
            plt.savefig(pdf, dpi=150)
            plt.close()
        else:
            plt.show()

    def plot_hot_spot(self, min_pos, max_pos, pdf=None):
        ''' plot mean firing rates on the grid of locations and indicate the
        corresponding preferred and non-preferred regions as determined by
        'define_hot_spot
        '''
        fig, ax = plt.subplots(nrows=1, ncols=2)
        plt.suptitle(self.filename + '_' + self.name, size=8)
        for irew in range(2):
            plt.sca(ax[irew])
            zmax = np.nanmax(self.frmean_space[irew,:,:])
            zmin = np.nanmin(self.frmean_space[irew,:,:])
            size = 50 * (self.frmean_space[irew,:,:] - zmin) / (zmax - zmin)
            print(size / 50 )
            plt.scatter(np.ravel(self.x_grid), np.ravel(self.y_grid),
                        s=np.ravel(size), c=self.rew_colors[irew])

            offset = 1
            bot_left = (self.x[min_pos[irew][0]] - offset,
                        self.y[min_pos[irew][2]] - offset)
            width = self.x[min_pos[irew][1]] - self.x[min_pos[irew][0]] + 2 * offset
            height = self.y[min_pos[irew][3]] - self.y[min_pos[irew][2]] + 2 * offset
            min_rect = plt.Rectangle(bot_left, width, height, facecolor='none',
                                     edgecolor='k', linestyle='dashed')
            plt.gca().add_patch(min_rect)
            bot_left = (self.x[max_pos[irew][0]] - offset,
                        self.y[max_pos[irew][2]] - offset)
            width = self.x[max_pos[irew][1]] - self.x[max_pos[irew][0]] + 2 * offset
            height = self.y[max_pos[irew][3]] - self.y[max_pos[irew][2]] + 2 * offset
            min_rect = plt.Rectangle(bot_left, width, height, facecolor='none',
                                     edgecolor='k')
            plt.gca().add_patch(min_rect)

            plt.xlabel('x (deg)')
            plt.ylabel('y (deg)')
        fig.tight_layout()
        if pdf:
            plt.savefig(pdf, dpi=150)
            plt.close()
        else:
            plt.show()

    def plot_gaussian(self, g, z, pdf=None):

        '''plot gaussian colormap with overlaid scatter of firing rates
           x, y are MxN array (typically denser than observed data points)
           g is a MxNxK array where K is the number of conditions
           z is actual firing rates
        '''
        fig, ax = plt.subplots(nrows=1, ncols=np.size(g,0))
        plt.suptitle(self.filename + '_' + self.name, size=8)
        for k in range(np.size(g,0)):

            #colormap
            plt.sca(ax[k])
            im = plt.imshow(np.transpose(g[k]), vmin=np.nanmin(g),
                            vmax=np.nanmax(g))
            plt.xticks((self.xy_max + self.x) * self.plot_density, self.x)
            plt.yticks((self.xy_max + self.y) * self.plot_density, self.y)
            plt.tick_params(labelsize=8)
            plt.xlabel('x (deg)')
            if k == 0:
                plt.ylabel('y (deg)')
            ax[k].invert_yaxis()

            #scatter
            x_scatter = (np.ravel(self.x_grid) + self.xy_max) * self.plot_density
            y_scatter = (np.ravel(self.y_grid) + self.xy_max) * self.plot_density
            size = np.ravel(z[k]) - np.nanmin(z[k])
            size = 50 * size / np.nanmax(size) + 10
            plt.scatter(x_scatter, y_scatter, s=size, c='w')

            divider = make_axes_locatable(ax[k])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.tick_params(labelsize=8)

        fig.tight_layout()
        if pdf:
            plt.savefig(pdf, dpi=150)
            plt.close()
        else:
            plt.show()

    ### ANALYSIS FUNCTIONS
    def define_hot_spot(self):
        ''' Define the region for which stimuli elicit the greatest firing
        rates (preferred region) or smallest firing rates (non-preferred
        region) for a neuron. Preferred/non-preferred regions may be any
        rectangular region encompassing locations on the grid. Non-preferred
        region is constrained to me non-overlapping with the preferred-region.

        -lb/ub: min/max size of the rectangluar region along each dimension
        '''
        lb = 1 # min size along dimension
        ub = 2 # max size along dimension (need to have ub < len(dim)//2)
        max_pos = [None, None]
        min_pos = [None, None]
        for irew in range(2):
            z = np.array(self.frmean_space[irew,:,:])
            z -= np.nanmean(z)
            curr_max = -np.inf
            for ix0 in range(len(self.x)-lb+1):
                for ix1 in range(ix0+lb-1, min([len(self.x), ix0+ub])):
                    for iy0 in range(len(self.y)-lb+1):
                        for iy1 in range(iy0+lb-1, min([len(self.y), iy0+ub])):
                            curr_val = np.nansum(z[ix0:ix1+1, iy0:iy1+1])
                            if curr_val > curr_max:
                                curr_max = curr_val
                                max_pos[irew] = (ix0, ix1, iy0, iy1)
            curr_min = np.inf
            for ix0 in range(len(self.x)-lb+1):
                for ix1 in range(ix0+lb-1, min([len(self.x), ix0+ub])):
                    for iy0 in range(len(self.y)-lb+1):
                        for iy1 in range(iy0+lb-1,  min([len(self.y), iy0+ub])):
                            if (ix0 > max_pos[irew][1] and ix1 > max_pos[irew][1]) or (
                                ix0 < max_pos[irew][0] and ix1 < max_pos[irew][0]) or (
                                iy0 > max_pos[irew][3] and iy1 > max_pos[irew][3]) or (
                                iy0 < max_pos[irew][2] and iy1 < max_pos[irew][2]):
                                curr_val = np.nansum(z[ix0:ix1+1, iy0:iy1+1])
                                if curr_val < curr_min:
                                    curr_min = curr_val
                                    min_pos[irew] = (ix0, ix1, iy0, iy1)

        if min_pos[0] == None or min_pos[1] == None:
            ipdb.set_trace()
        return min_pos, max_pos

    ### I/O

    def save_neuron(self):
        ''' Create a pickle of the neuron object '''
        fname = self.filename + '_' + self.name
        print('saving:', self.directory + fname)
        with open(self.directory + fname + '.pickle', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
