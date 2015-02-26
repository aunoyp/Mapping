# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:46:48 2015

@author: cjpeck
"""

import pdb
import imp
import sys
import time
import numpy as np
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt


import experiment
#if 'experiment' in sys.modules:
#    imp.reload(experiment)
#else:
#    import experiment
#from experiment import Experiment, Neuron
    
def flattened_conv(x, binSize, binShift):
    
    M = np.size(x,0)
    N = np.size(x,1)
    if binSize <= 0 or binSize > N:
        return
    
    # need to get rid of NaNs before passing to 'fftconvolve'
    # keep track of their location, so they can be re-NaNed later
    nan_inds = np.isnan(x)        
    # flattten and convolve
    x_flat = np.array(x)
    x_flat[nan_inds] = 0    
    x_flat = x_flat.ravel()        
    x_conv = sp.signal.fftconvolve(x_flat, 
                                   np.ones((binSize,)) / binSize, mode='valid')
        
    x_out = np.full((M, N - (binSize-1)), np.nan)
    i = 0 #indice in flattened array
    row = 0 #row in out put matrix
    while i < len(x_conv):        
        x_out[row,:] = x_conv[i : i + (N - (binSize - 1))]
        i += N
        row += 1    
    
    shifted = np.array(nan_inds)
    for i in range(1, binSize):        
        shifted = np.hstack((shifted[:,1:], np.full((M,1), False, dtype=bool)))
        nan_inds = nan_inds | shifted
    nan_inds = nan_inds[:, :N-(binSize-1)]
    x_out[nan_inds] = np.nan

    return x_out[:, ::binShift]


def runningMean(x, binSize, binShift):    
    return np.convolve(x, np.ones((binSize,)) / binSize, 
                       mode='valid')[::binShift]   


if __name__ == '__main__':
    
    finfo = sp.io.loadmat(
        '/Users/cjpeck/Dropbox/Matlab/custom offline/mapping/files/' + 
        'map_cell_list.mat', squeeze_me=True)
    finfo['cell_ind'] -= 1
    finfo['file_ind'] -= 1        
    
    directory = '/Users/cjpeck/Documents/Matlab/Blackrock/Data/MAP_PY/'    
    file = finfo['filenames'][0]
    #f = sp.io.loadmat(directory + file + '.nex.mat', 
    #              squeeze_me=True, struct_as_record=True)
    cells = finfo['cell_name'][finfo['file_ind'] == 0]  
    
    neuron = experiment.Neuron(f, file, cells[2])
    
    # test data
    #a = np.array([[1,2,3,np.nan], [np.nan, 4,5,6], [7,8,np.nan,9]])
    #for i in range(2,3):
    #    print(flattened_conv(a, i))
        
    # smooth data for one neurons
    binSize = 4
    binShift = 2
    start = time.time()
    new_fr = flattened_conv(neuron.fr, binSize, binShift)
    print(time.time() - start)
    ind = 8
    #print(np.c_[neuron.fr[ind,:], np.r_[new_fr[ind,:], np.array([None] * (binSize - 1))]])
    
    # get new firing rate bin times
    tStart = runningMean(neuron.tStart, binSize, binShift)
    tEnd = runningMean(neuron.tEnd, binSize, binShift)
    print(np.c_[tStart, tEnd])
    
    t0 = np.mean(np.c_[neuron.tStart, neuron.tEnd],1)
    t1 = np.mean(np.c_[tStart, tEnd],1)
    fr0 = np.nanmean(neuron.fr, 0)
    fr1 = np.nanmean(new_fr, 0)
    plt.figure()
    #plt.subplot(121)
    plt.plot(t0, fr0)
    #plt.subplot(122)
    plt.plot(t1, fr1)
    plt.show()


