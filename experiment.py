# -*- coding: utf-8 -*-

import scipy as sp
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import pickle
import pdb

#   Experiment:
#       files - list of files in that experiment
#   Session:
#       neurons - list of neurons in that session
#       analog data
#   Neuron
#       df (DataFrame) - trial info & spikes
#       firing rates


# WANT TO CHANGE DATA FRAME TO A SIMPLE DICITONARY WITH NP.ARRAYS
# -easier for indexing 

class Experiment(object):
    
    def __init__(self, OVERWRITE=False):
        self.files = OrderedDict()
        self.OVERWRITE = OVERWRITE
            
    def add_session(self, f, fname, cells):       
        
        if len(cells)==0: 
            print(fname, ' has no cells?') 
            return
        if not fname in self.files or self.OVERWRITE:
            self.files[fname] = cells
            session = Session(f, fname, cells, self.OVERWRITE)
            session.save_session()
            
    def save_experiment(self):
        directory = '/Users/cjpeck/Dropbox/Matlab/custom offline/mapping_py/mapping/data/'
        print('saving: mapping_exp')
        with open(directory + 'mapping_exp.pickle', 'wb') as f:        
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
                                    
class Session(object):
    
    def __init__(self, f, fname, cells, OVERWRITE=False):
        self.neurons = OrderedDict()
        self.filename = fname
        self.OVERWRITE = OVERWRITE        
        for cell in cells:
            self.add_neuron(f, fname, cell)
        self.process_analog_data(f)
        
    def add_neuron(self, f, fname, cellname):
        cell_fname = fname + '_' + cellname
        if not cell_fname in self.neurons or self.OVERWRITE:
            self.neurons[cell_fname] = True
            neuron = Neuron(f, fname, cellname)
            neuron.save_neuron()
            
    def process_analog_data(self, f):
        pass
    
    def save_session(self):
        directory = '/Users/cjpeck/Dropbox/Matlab/custom offline/mapping_py/mapping/data/'
        print('saving:', self.filename)
        with open(directory + self.filename + '.pickle', 'wb') as f:        
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    
class Neuron(object):
    
    def __init__(self, f, fname, cellname):    
        
        # cell info
        self.filename = fname
        self.name = cellname        
        self.df = None
        
        # parameters for firing rates
        self.tFrame = np.array([-500, 1500])
        self.tInt = 10
        self.tShift = 10
        self.tStart = np.linspace(self.tFrame[0], self.tFrame[1] - self.tInt, 
                                  (self.tFrame[1] - self.tFrame[0]) / self.tShift)
        self.tEnd = np.linspace(self.tFrame[0] + self.tInt, self.tFrame[1], 
                                (self.tFrame[1] - self.tFrame[0]) / self.tShift)
        
        # covert to seconds
        self.tFrame /= 1e3
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

        # need something to say whehter it is MUA or SUA
        if self.name[-2:] == 'wf':
            print('NEURON HAS _WF SUFFIX')
        self.SUA = self.name[-1] != 'U'
        
        # which side is contra
        tmp = f['dat'][()]['data']['CUE_X'][f['dat'][()]['data']['CUE_CONTRA'] == 1]
        tmp = np.unique(np.sign(tmp, dtype=int))
        if len(tmp)==1:
            self.contra = tmp[0]
        else:
            print('cant determine which side is conta')
            pdf.set_trace()

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
        
        # need the indice of the cell in that file
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
        
    def get_data_labels(self):
        labels = [('spkdata', 'float'), 
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
        return labels

        
    def extract_trial_info(self, f, cell_ind, labels):
        
        # trials where the recording of this neuron is valid
        # ONLY these trials will be saved
        good_trial = np.where(
                    [f['dat'][()]['data'][()]['good_trial'][i][cell_ind]==True 
                    for i in range(self.ntrials)])[0]

        # DICT with all trial information
        # all information is the same (one per trial) EXCEPT for 'spkdata' 
        # which can be any number of elements per trial

        ### these are all CAST as OBJECTS - need to specify int or float
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
                                
    def smooth_firing_rates(self, binSize, binShift):

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
        return np.convolve(t, np.ones((binSize,)) / binSize, 
                           mode='valid')[::binShift]   
    
    def psth(self):
        if self.fr_smooth == None:
            self.smooth_firing_rates(10, 1)
        t0 = np.mean(np.c_[self.tStart, self.tEnd],1)
        t1 = np.mean(np.c_[self.tStart_smooth, self.tEnd_smooth], 1)
        fr0 = np.nanmean(self.fr, 0)
        fr1 = np.nanmean(self.fr_smooth, 0)
        plt.figure()
        plt.plot(t0, fr0)
        plt.plot(t1, fr1)
        plt.show()
        
    def psth_rew(self):
        if self.fr_smooth == None:
            self.smooth_firing_rates(10, 1)            
        t = np.mean(np.c_[self.tStart_smooth, self.tEnd_smooth], 1)      
        fr0 = np.nanmean(self.fr_smooth[np.array(self.df['rew']==0),:], 0)
        fr1 = np.nanmean(self.fr_smooth[np.array(self.df['rew']==1),:], 0)
        plt.plot(t, fr0, 'r')
        plt.plot(t, fr1, 'b')
        plt.show()        
        
    def save_neuron(self):        
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
    neuron = Neuron(f, finfo['filenames'][0], cells[1])
    
    io = LoadData()
    
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
                          
    # add neuron by file  
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
    
    
    
    
        