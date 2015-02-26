# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:46:48 2015

@author: cjpeck
"""

import pdb
import imp
import time
import numpy as np
import scipy as sp

if 'experiment' in sys.modules:
    imp.reload(experiment)
else:
    import experiment
#from experiment import Experiment, Neuron
    
# safe and slow methods, looks at all spikes on each iteration
def get_fr1(neuron):
    fr = np.full((neuron.df.shape[0], len(neuron.tStart)), np.nan)
    for ind, spks in neuron.df['spkdata'].iteritems():
        cueOn = neuron.df['t_CUE_ON'].loc[ind]
        if not np.isnan(cueOn):
            end_time = np.nanmin([neuron.df['t_TARG_ON'].loc[ind] - cueOn,
                            neuron.df['t_BREAK_TARG'].loc[ind] - cueOn,
                            neuron.df['t_BREAK_CUE'].loc[ind] - cueOn,
                            neuron.df['t_TRIAL_END'].loc[ind] - cueOn])
            for t in range(len(neuron.tStart)):
                if end_time < neuron.tEnd[t]:
                    break
                nspks = np.sum(np.bitwise_and(spks - cueOn >= neuron.tStart[t], 
                                       spks - cueOn < neuron.tEnd[t]))
                fr[ind, t] = nspks / neuron.tInt
    return fr
      
# need to modify for overlapping bins
def get_fr2(neuron):
    fr = np.full((neuron.df.shape[0], len(neuron.tStart)), np.nan)
    for ind, spks in neuron.df['spkdata'].iteritems():
        cueOn = neuron.df['t_CUE_ON'].loc[ind]
        if not np.isnan(cueOn):                        
            start_ind = 0
            end_time = np.nanmin([neuron.df['t_TARG_ON'].loc[ind] - cueOn,
                            neuron.df['t_BREAK_TARG'].loc[ind] - cueOn,
                            neuron.df['t_BREAK_CUE'].loc[ind] - cueOn,
                            neuron.df['t_TRIAL_END'].loc[ind] - cueOn])
            for t in range(len(neuron.tStart)):  
                # if time frame is after an 'end' event, fr = nan          
                if end_time < neuron.tEnd[t]:
                    break
                # if at end of spike list, fr = 0 
                if start_ind == len(spks):
                    fr[ind, t] = 0
                    continue
                # counts spikes in this time window
                curr_ind = start_ind
                nspks = 0
                while (curr_ind < len(spks) and 
                       spks[curr_ind] - cueOn < neuron.tEnd[t]):
                    if spks[curr_ind] - cueOn >= neuron.tStart[t]:
                        nspks +=1
                    curr_ind +=1     
                # compue firing rate
                fr[ind, t] = nspks / neuron.tInt
                # set the start point for searching for spikes to be first
                # spike after the current time window
                start_ind = curr_ind                                                    
                
                
                    
    return fr
    


if __name__ == '__main__':

    finfo = sp.io.loadmat(
        '/Users/cjpeck/Dropbox/Matlab/custom offline/mapping/files/' + 
        'map_cell_list.mat', squeeze_me=True)
    finfo['cell_ind'] -= 1
    finfo['file_ind'] -= 1        
    
    directory = '/Users/cjpeck/Documents/Matlab/Blackrock/Data/MAP_PY/'    
    file = finfo['filenames'][0]
    f = sp.io.loadmat(directory + file + '.nex.mat', 
                  squeeze_me=True, struct_as_record=True)
    cells = finfo['cell_name'][finfo['file_ind'] == 0]    
    
    total1 = 0
    total2 = 0
    for cell in cells:
        
        neuron = experiment.Neuron(f, file, cell)
        print(cell)
    
        start = time.time()
        #fr1 = get_fr1(neuron)
        elapsed = time.time() - start
        total1 += elapsed
        print(' Method 1:', elapsed)
        
        start = time.time()
        fr2 = get_fr2(neuron)
        elapsed = time.time() - start
        total2 += elapsed
        print(' Method 2:', elapsed)
        
#        nans_are_equal = not np.sum(np.isnan(fr1) != np.isnan(fr2))    
#        fr1 = np.ravel(fr1)
#        fr2 = np.ravel(fr2)
#        fr1 = fr1[np.isnan(fr1) == False]
#        fr2 = fr2[np.isnan(fr2) == False]
#        tol = 1e-6
#        is_equal = np.bitwise_and(fr1 >= fr2 - tol, fr1 <= fr2 + tol)
#        floats_are_equal = not np.sum(is_equal == False)
#    
#        if not (nans_are_equal and floats_are_equal):
#            pdb.set_trace()
        
print('Method 1 total:', total1)
print('Method 2 total:', total2)
