# -*- coding: utf-8 -*-

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
import pickle
import scipy as sp
import scipy.signal


'''
Git command line:

git add 'experiment.py'
git commit -m 'updated experiment.py'
git remote add origin https://github.com/cjpeck/Mapping.git
git push -u origin master

General structure:
   Experiment:
       files - list of files in that experiment
   Session:
       neurons - list of neurons in that session
       analog data
   Neuron
       df (DataFrame) - trial info & spikes
       firing rates
'''

class Experiment(object):    
    '''Experiment object: Maintains dictionary of experiment sessions and 
    corresponding neurons.'''    
    def __init__(self, OVERWRITE=False):
        ''' Initialize dictionary'''
        self.files = OrderedDict()
        self.OVERWRITE = OVERWRITE
            
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
        directory = '/Users/cjpeck/Dropbox/Matlab/custom offline/mapping_py/mapping/data/'
        print('saving: mapping_exp')
        with open(directory + 'mapping_exp.pickle', 'wb') as f:        
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
                                    
class Session(object):
    '''Session object: Maintains list of cells in that session and contains
    analog behavioral data relevent to all cells in that session.
    '''        
    def __init__(self, f, fname, cells, OVERWRITE=False):
        ''' Initialize list of cells names.'''
        self.neurons = []
        self.filename = fname
        self.ntrials = len(f['dat'][()]['data'])
        self.OVERWRITE = OVERWRITE        
        for cell in cells:
            self.add_neuron(f, fname, cell)
        self.eyes = []
        self.laser = []
        self.pupil = []
        self.process_analog_data(f)
        
    def add_neuron(self, f, fname, cellname):
        '''Append neuron names to list of neurons in that session. Create
        'neuron' objects for all neurons in that session. 
        
        if neuron is already in the list, do nothing unless 
        self.OVERWRITE = True
        
        Neuron in the names are in the format 'session_name' + '_' + 
        'neuron_name' in order to uniquely name all neuron objects
        '''
        cell_fname = fname + '_' + cellname
        if not cell_fname in self.neurons or self.OVERWRITE:
            self.neurons.append(cell_fname)
            neuron = Neuron(f, fname, cellname)
            neuron.save_neuron()
            
    def process_analog_data(self, f):
        '''Clean behavoral analog data inclding eye position, laser, and
        pupil diameter
        '''                
        for i in range(self.ntrials):
            # checks to make sure the sampling frequency is 1000hz 
            dt = (f['dat'][()]['data']['eyes_dt'][i], 
                  f['dat'][()]['data']['laser_dt'][i],
                  f['dat'][()]['data']['pupil_dt'][i])                     
            if sum([abs(x - 0.001) > 10e-6 for x in dt]):                
                print('analog dt is not 1000hz')
                pdb.set_trace()        
            # check that the start time of analog data collection is about 0,
            # pad with NaNs if start is slightly after zero
            t0 = (f['dat'][()]['data']['eyes_start_t'][i], 
                  f['dat'][()]['data']['laser_start_t'][i],
                  f['dat'][()]['data']['pupil_start_t'][i])                   
            nans_to_add = {round(x / 1e-3) for x in t0}            
            if len(nans_to_add) == 1:
                nans_to_add = nans_to_add.pop()
            else:
                print('different t0 for each analog data type')
                pdb.set_trace()
            if nans_to_add > 5:
                print('analog start time is > 5 ms')
                pdb.set_trace()
    
            self.eyes.append(np.vstack((np.full((nans_to_add,2), np.nan), 
                                        np.array(f['dat'][()]['data']['eyes'][i], 
                                        dtype=float))))
            self.laser.append(np.hstack((np.full((nans_to_add,), np.nan), 
                                        np.array(f['dat'][()]['data']['laser'][i], 
                                        dtype=float)))) 
            self.pupil.append(np.hstack((np.full((nans_to_add,), np.nan), 
                                        np.array(f['dat'][()]['data']['pupil'][i], 
                                        dtype=float))))                                        
            
    def save_session(self):
        '''Save session object'''
        directory = '/Users/cjpeck/Dropbox/Matlab/custom offline/mapping_py/mapping/data/'
        print('saving:', self.filename)
        with open(directory + self.filename + '.pickle', 'wb') as f:        
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    
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
    
    '''
    def __init__(self, f, fname, cellname):    
        
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
        self.tEnd = np.linspace(self.tFrame[0] + self.tInt, self.tFrame[1], 
                                (self.tFrame[1] - self.tFrame[0]) / self.tShift)
        
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

        # need something to say whether neuron is MUA or SUA
        if self.name[-2:] == 'wf':
            print('NEURON HAS _WF SUFFIX')
        self.SUA = self.name[-1] != 'U'
        
        # which side of the screen is contralateral to the recording site?
        # left: -1, right: 1
        tmp = f['dat'][()]['data']['CUE_X'][f['dat'][()]['data']['CUE_CONTRA'] == 1]
        tmp = np.unique(np.sign(tmp, dtype=int))
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
        self.fr = np.full((self.df.shape[0], len(self.tStart)), np.nan)
        for ind, spks in self.df['spkdata'].iteritems():
            cueOn = self.df['t_CUE_ON'].loc[ind]
            if not np.isnan(cueOn):                        
                start_ind = 0
                end_time = np.nanmin([self.df['t_TARG_ON'].loc[ind] - cueOn,
                                self.df['t_BREAK_TARG'].loc[ind] - cueOn,
                                self.df['t_BREAK_CUE'].loc[ind] - cueOn,
                                self.df['t_TRIAL_END'].loc[ind] - cueOn])
                for t in range(len(self.tStart)):  
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
                    self.fr[ind, t] = nspks / self.tInt
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
        x, y = self.get_xy()
        self.fr_space = np.empty((2, len(self.tStart_smooth), len(x), len(y)))
        for ix in range(len(x)):
            for iy in range(len(y)):
                for irew in range(2):
                    inds = self.get_xy_rew_inds(x[ix], y[iy], irew)           
                    self.fr_space[irew, :, ix, iy] = np.nanmean(self.fr_smooth[inds,:], 0)
    
    def get_frmean_by_loc(self, tMean):
        ''' Computes the mean firing rate for each spatial location '''
        x, y = self.get_xy()
        t = self.get_mean_t(tMean)
        self.frmean_space = np.empty((2, len(x), len(y)))
        for ix in range(len(x)):
            for iy in range(len(y)):
                for irew in range(2):
                    inds = self.get_xy_rew_inds(x[ix], y[iy], irew)   
                    self.frmean_space[irew, ix, iy] = \
                        np.nanmean(np.mean(self.fr[np.ix_(inds, t)], 1))
                    
    ### AUXILLARY FUNCTIONS
    
    def get_xy(self):
        ''' Return all possible x and y locations of the stimuli'''
        x = np.unique(self.df['CUE_X'])
        x = x[np.isnan(x)==False]
        y = np.unique(self.df['CUE_Y'])
        y = y[np.isnan(y)==False]
        return x, y
        
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
        
    def psth_map(self, binSize=10, binShift=1):
        ''' Firing rates relative to cue onset for each reward condition 
        (reward or no reward) and spatial location
        '''
        if self.fr_smooth == None:
                self.smooth_firing_rates(binSize, binShift)     
        x, y = self.get_xy()                       
        self.get_fr_by_loc()
        ylim = self.get_ylim()
        
        t = np.mean(np.c_[self.tStart_smooth, self.tEnd_smooth], 1)
        fig, ax = plt.subplots(nrows=len(y), ncols=len(x))
        for xi in range(len(x)):
            for yi in range(len(y)):
                #plot
                plt.sca(ax[len(y) - yi - 1, xi])  
                plt.plot(t, self.fr_space[0, :, xi, yi], color='r')
                plt.plot(t, self.fr_space[1, :, xi, yi], color='b')            
                plt.plot((0,0), ylim, linestyle='--', color='0.5')
                #format
                plt.title('x=%1.1f, y=%1.1f' % (x[xi], y[yi]), size=6)
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
        
        #plt.savefig(title + '_psth_map.eps', bbox_inches='tight')
        plt.show()    
        
    def psth_rew(self, binSize=10, binShift=1):
        ''' Firing rates relative to cue onset for cue predicted reward and
        no reward (irrespective of spatial location)
        '''
        if self.fr_smooth == None:
            self.smooth_firing_rates(binSize, binShift)            
        t = np.mean(np.c_[self.tStart_smooth, self.tEnd_smooth], 1)      
        fr0 = np.nanmean(self.fr_smooth[np.array(self.df['rew']==0),:], 0)
        fr1 = np.nanmean(self.fr_smooth[np.array(self.df['rew']==1),:], 0)
        plt.plot(t, fr0, 'r')
        plt.plot(t, fr1, 'b')
        plt.show()   
        
    def plot_hot_spot(self, min_pos, max_pos):
        ''' plot mean firing rates on the grid of locations and indicate the
        corresponding preferred and non-preferred regions as determined by
        'define_hot_spot
        '''
        x, y = self.get_xy()
        fig, ax = plt.subplots(nrows=1, ncols=2)
        plt.suptitle(self.filename + '_' + self.name, size=8)
        for irew in range(2):
            plt.sca(ax[irew])
            zmax = np.nanmax(neuron.frmean_space[irew,:,:])
            zmin = np.nanmin(neuron.frmean_space[irew,:,:])
            size = 50 * (self.frmean_space[irew,:,:] - zmin) / (zmax - zmin)            
            x_grid = [xi for xi in x for _ in y]
            y_grid = [yi for _ in x for yi in y]            
            plt.scatter(x_grid, y_grid, s=np.ravel(size), c=self.rew_colors[irew])
            
            offset = 1
            bot_left = (x[min_pos[irew][0]] - offset, 
                        y[min_pos[irew][2]] - offset)
            width = x[min_pos[irew][1]] - x[min_pos[irew][0]] + 2 * offset
            height = y[min_pos[irew][3]] - y[min_pos[irew][2]] + 2 * offset
            min_rect = plt.Rectangle(bot_left, width, height, facecolor='none',
                                     edgecolor='k', linestyle='dashed')
            plt.gca().add_patch(min_rect) 
            bot_left = (x[max_pos[irew][0]] - offset, 
                        y[max_pos[irew][2]] - offset)
            width = x[max_pos[irew][1]] - x[max_pos[irew][0]] + 2 * offset
            height = y[max_pos[irew][3]] - y[max_pos[irew][2]] + 2 * offset
            min_rect = plt.Rectangle(bot_left, width, height, facecolor='none',
                                     edgecolor='k')
            plt.gca().add_patch(min_rect) 
            
            plt.xlabel('x (deg)')
            plt.ylabel('y (deg)') 
        fig.tight_layout()
        plt.show() 
        
    ### ANALYSIS FUNCTIONS

    def define_hot_spot(self, tMean):
        ''' Define the region for which stimuli elicit the greatest firing 
        rates (preferred region) or smallest firing rates (non-preferred
        region) for a neuron. Preferred/non-preferred regions may be any 
        rectangular region encompassing locations on the grid. Non-preferred 
        region is constrained to me non-overlapping with the preferred-region. 
        
        -tMean: time window of firing rates for which to perform this analysis.
        -lb: minimimum size of the rectangluar region along each dimension        
        '''
        self.get_frmean_by_loc(tMean)
        x, y = self.get_xy()
        lb = 2 # min size along dimension
        ub = 3 # max size along dimension
        max_pos = [None, None]
        min_pos = [None, None]
        for irew in range(2):
            z = np.array(self.frmean_space[irew,:,:])
            z -= np.nanmean(z)
            curr_max = -np.inf
            for ix0 in range(len(x)):
                for ix1 in range(ix0+lb-1, len(x)):
                    for iy0 in range(len(y)):
                        for iy1 in range(iy0+lb-1, len(y)):
                            curr_val = np.nansum(z[ix0:ix1+1, iy0:iy1+1])
                            if curr_val > curr_max:
                                curr_max = curr_val
                                max_pos[irew] = (ix0, ix1, iy0, iy1)
            curr_min = np.inf
            for ix0 in range(len(x)):
                for ix1 in range(ix0+lb-1, len(x)):
                    for iy0 in range(len(y)):
                        for iy1 in range(iy0+lb-1, len(y)):
                            print((ix0, ix1, iy0, iy1))
                            if (ix0 > max_pos[irew][1] and ix1 > max_pos[irew][1]) or (
                                ix0 < max_pos[irew][0] and ix1 < max_pos[irew][0]) or (
                                iy0 > max_pos[irew][3] and iy1 > max_pos[irew][3]) or (
                                iy0 < max_pos[irew][2] and iy1 < max_pos[irew][2]):                            
                                curr_val = np.nansum(z[ix0:ix1+1, iy0:iy1+1])
                                if curr_val < curr_min:
                                    curr_min = curr_val
                                    min_pos[irew] = (ix0, ix1, iy0, iy1)
                                    
        if min_pos[0] == None or min_pos[1] == None:
            pdb.set_trace()
        return min_pos, max_pos
                                                                          
    ### I/O
        
    def save_neuron(self):    
        ''' Create a pickle of the neuron object '''
        directory = '/Users/cjpeck/Dropbox/Matlab/custom offline/mapping_py/mapping/data/'
        fname = self.filename + '_' + self.name
        print('saving:', directory + fname)
        with open(directory + fname + '.pickle', 'wb') as f:        
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


class LoadData(object):
            
    def __init__(self):
        self.directory = '/Users/cjpeck/Dropbox/Matlab/custom offline/mapping_py/mapping/data/'
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
        



if __name__ == '__main__':
    
    # initialize Experiment, and load information    
    finfo = sp.io.loadmat(
        '/Users/cjpeck/Dropbox/Matlab/custom offline/mapping/files/' + 
        'map_cell_list.mat', squeeze_me=True)
        
    # changed to zero based inds
    finfo['cell_ind'] -= 1
    finfo['file_ind'] -= 1        

    # test data
    directory = '/Users/cjpeck/Documents/Matlab/Blackrock/Data/MAP_PY/'
    f = sp.io.loadmat(directory + finfo['filenames'][0] + '.nex.mat', 
                      squeeze_me=True, struct_as_record=True)
    cells = finfo['cell_name'][finfo['file_ind'] == 0]
    session = Session(f, finfo['filenames'][0], cells[0], True)
    neuron = Neuron(f, finfo['filenames'][0], cells[0])

    io = LoadData()
    for file in io.experiment.files:
        for cell in io.experiment.files[file]:
            neuron = io.load_neuron(file, cell)
            neuron.psth_map()    
            min_pos, max_pos = neuron.define_hot_spot((.1, .5))
            neuron.plot_hot_spot(min_pos, max_pos)                
    
    # test smooth firing rates procedure    
#    neuron.smooth_firing_rates(10, 1)    
#    t0 = np.mean(np.c_[neuron.tStart, neuron.tEnd],1)
#    t1 = np.mean(np.c_[neuron.tStart_smooth, neuron.tEnd_smooth],1)
#    fr0 = np.nanmean(neuron.fr, 0)
#    fr1 = np.nanmean(neuron.fr_smooth, 0)
#    plt.figure()
#    plt.plot(t0, fr0)
#    plt.plot(t1, fr1)
#    plt.show()
                          
#    # add neuron by file  
#    overwrite = True
#    exp = Experiment(overwrite)
#    for iFile, file in enumerate(finfo['filenames']):
#        # load file
#        f = sp.io.loadmat(directory + file + '.nex.mat', squeeze_me=True)
#        print('Loaded file', file)
#        cells = finfo['cell_name'][finfo['file_ind'] == iFile]             
#        exp.add_session(f, file, cells)        
#    exp.save_experiment()              
        
#    # a test
#    n = Neuron(f, file, cells[0])
#    for iTrial in range(60):
#        if iTrial in n.df.index:
#            print(iTrial, len(n.df['spkdata'].loc[iTrial]), 'spikes')
#        else:
#            print(iTrial, 'not in DataFrame')

#    # Query time
#    s = set()
#    l = []
#    a = np.full(np.shape(finfo['cell_name']), 'nan', dtype=object)
#    for i, name in enumerate(finfo['cell_name']):
#        fname = finfo['filenames'][finfo['file_ind'][i]] + '_' + name
#        s.add(fname)
#        l.append(fname)
#        a[i] = fname
#    
#    tests = ['tn_map_120314_elec1U', 'tn_map_120914_elec21U', 'tn_map_123014_elec31b']
#    t_set = [0]*len(tests)
#    t_list = [0]*len(tests)
#    t_array = [0]*len(tests)
#    for t in range(len(tests)):
#        start = time.time()
#        tests[t] in s
#        t_set[t] = time.time() - start
#        start = time.time()
#        tests[t] in l
#        t_list[t] = time.time() - start
#        start = time.time()
#        tests[t] in a
#        t_array[t] = time.time() - start
#    print('set:', t_set)
#    print('list:', t_list)
#    print('array:', t_array)
    
#    io = LoadData()
#    fsession0 = io.load_session('tn_map_123014')
#    print(fsession0.filename)
#    fsession = io.load_session('asdf')
#    print(fsession.filename)
#    fneuron0 = io.load_neuron('tn_map_123014', 'elec1U')
#    print(fneuron0.filename, fneuron0.name)
#    fneuron = io.load_neuron('aasdf', 'asdf')
#    print(fneuron.filename, fneuron.name)
    
    
    
    
        