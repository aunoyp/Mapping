# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:21:23 2015

@author: cjpeck
"""

from collections import OrderedDict
import copy
from itertools import combinations  
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import pdb
import pickle
import scipy as sp
import scipy.io
import scipy.optimize

from sklearn import linear_model

from experiment4 import Neuron, LoadData

# modules to add
from matplotlib.backends.backend_pdf import PdfPages

'''
-beta0 is the initial params and does not change, operate on betas only
'''

class ExNeuron(object):
    
    def __init__(self, neuron):
        ''' Just copy all the attributed from the original pickled 'Neuron', 
        but am redefining all the methods with this class.
        '''    
        for attr in neuron.__dict__:
            self.__setattr__(attr, neuron.__getattribute__(attr))
        self.betas0 = copy.copy(self.betas)        
         
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
        
    def psth_map(self, pdf):
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
        
        pdf.savefig()
        plt.close()              
                               
    def plot_gaussian(self, g, z, pdf, title):
        
        '''plot gaussian colormap with overlaid scatter of firing rates
           x, y are MxN array (typically denser than observed data points)
           g is a MxNxK array where K is the number of conditions
           z is actual firing rates
        '''
        fig, ax = plt.subplots(nrows=1, ncols=np.size(g,0))
        plt.suptitle(title, size=8)
        
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
        pdf.savefig()
        plt.close()
            
    def plot_values(self):
        g_nr = self.get_gauss(self.x_plot_grid, self.y_plot_grid, 0, 
                                            self.betas)
        g_rw = self.get_gauss(self.x_plot_grid, self.y_plot_grid, 1, 
                                self.betas)        
        mse = self.error_func(self.betas)
        return g_nr, g_rw, mse
    
    def get_ylim(self):
        ''' For plotting, find the max and min firing rates across all
        spatial locations and reward conditions
        '''
        ymin = int(np.floor(np.nanmin(self.fr_space)))
        ymax = int(np.ceil(np.nanmax(self.fr_space)))
        return (ymin, ymax)
        
    def basinhop(self, niter=100, method='Nelder-Mead', print_betas=True):
        results = sp.optimize.basinhopping(self.error_func, self.betas0, 
                                           minimizer_kwargs={'method': method}, 
                                           disp=False, niter=niter)
        self.betas = results.x
        if print_betas:
            print('\tSTART\tFINISH')
            for i in range(len(self.betas0)):
                print('%s:\t%.2f\t%.2f' % (self.param_labels[i], 
                                           self.betas0[i], self.betas[i]))
            print('MSE:\t%.2f\t%.2f' %(self.error_func(self.betas0), self.error_func(self.betas)))
        
    def minimize(self, method='Nelder-Mead', bounds=None, print_betas=True):
        results = sp.optimize.minimize(self.error_func, self.betas0, method=method, bounds=bounds)
        self.betas = results.x
        print('\tSTART\tFINISH')
        for i in range(len(self.betas0)):
            print('%s:\t%.2f\t%.2f' % (self.param_labels[i], 
                                       self.betas0[i], self.betas[i]))
        print('MSE:\t%.2f\t%.2f' %(self.error_func(self.betas0), self.error_func(self.betas)))
    
def get_file_info():
    # initialize Experiment, and load information    
    finfo = sp.io.loadmat(
        '/Users/cjpeck/Dropbox/Matlab/custom offline/mapping/files/' + 
        'map_cell_list.mat', squeeze_me=True)        
    # changed to zero based inds
    finfo['cell_ind'] -= 1
    finfo['file_ind'] -= 1      
    return finfo
    
def load_neuron_get_info(file, cell):
    print('Loading neuron', file, cell)
    neuron = io.load_neuron(file, cell)
    neuron.get_xy()
    neuron.get_xy_grid()
    neuron.get_xy_plot_grid()    
    neuron.get_frmean_by_loc((.1,.5))    
    neuron.get_initial_params()
    neuron.smooth_firing_rates(10,1)
    neuron.get_fr_by_loc()
    return neuron

'''
cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
...         {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
...         {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
'''
    
if __name__ == '__main__':
    
    param_labels = ['ux', 'uy', 'stdx', 'stdy',
                    'minfr_nr', 'maxfr_nr', 'minfr_rw', 'maxfr_rw']    
    
    '''
    #sucks: 'Powell'
    #needs jacobian: 'Newton-CG', 'COBYLA', 'dogleg'
    methods = ['Nelder-Mead', 'CG', 'BFGS', 
               'L-BFGS-B', 'TNC', 'SLSQP']
               
    #bounds = ((-20, 20), (-20, 20), (0,40), (0,40),
    #          (0, 150), (0, 200), (0,150), (0,200))    
    io = LoadData()
    niter = 100
    fig_dir = '/Users/cjpeck/Dropbox/Matlab/custom offline/' + \
              'mapping_py/mapping/figs/'       
    
    params = {}
    for method in methods:
        params[method] = []
    for file in io.experiment.files:
        for cell in io.experiment.files[file]:   
            neuron0 = load_neuron_get_info(file, cell)            
            neuron = ExNeuron(neuron0)
            for method in methods:
                neuron.basinhop(niter=niter, method=method, print_betas=False)
                mse = neuron.error_func(neuron.betas)
                mse0 = neuron.error_func(neuron.betas0)
                params[method].append((mse, neuron.betas))        
    directory = '/Users/cjpeck/Dropbox/Matlab/custom offline/mapping_py/mapping/data/'
    fname = 'model_testng'
    print('saving:', directory + fname)
    with open(directory + fname + '.pickle', 'wb') as f:        
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)
        
    x = np.array([[x[0] for x in params[key]] for key in params]).transpose()
    mses = pd.DataFrame(x, columns=methods)
    mses.plot()
    '''
    
    '''    
    files = {'tn_map_120314': ['elec19U', 'elec31U'],
             'tn_map_120414': ['elec14a', 'elec15a', 'elec17b', 'elec19U', 'elec19a'],
             'tn_map_120514': [], 
             'tn_map_120814': ['elec19b', 'elec29U'],
             'tn_map_120914': ['elec3U', 'elec5U', 'elec5a', 'elec5b', 'elec10a', 'elec14b', 'elec21a'],
             'tn_map_121114': ['elec1U', 'elec1a', 'elec6a'], 
             'tn_map_121214': ['elec1U', 'elec2U', 'elec25a', 'elec25b'], 
             'tn_map_121514': []}            
    for file in io.experiment.files:
        for cell in io.experiment.files[file]:            
            neuron0 = load_neuron_get_info(file, cell)            
            neuron = ExNeuron(neuron0)
            with PdfPages(fig_dir + file + '_' + cell + '_testing.pdf') as pdf:
                
                neuron.psth_map(pdf)                     
                for method in methods:
                    print('\n*** MINIMIZE ' + method)                
                    neuron.minimize(method=method, bounds=None)
                    g_nr, g_rw, mse = neuron.plot_values()
                    title = '%s %s %.2f MINIMIZE %s' % (file, cell, mse, method)
                    neuron.plot_gaussian([g_nr, g_rw], neuron.frmean_space, pdf, title)
    '''


    directory = '/Users/cjpeck/Dropbox/Matlab/custom offline/mapping_py/mapping/data/'
    fname = 'model_testng'
    print('saving:', directory + fname)
    with open(directory + fname + '.pickle', 'rb') as f:        
        params = pickle.load(f)
        
    df_means = pd.DataFrame(columns=['mse'] + param_labels)
    df_nans = pd.DataFrame(columns=['mse'] + param_labels)
    for method in params:
        mses = np.array([x[0] for x in params[method]])
        betas = np.array([x[1] for x in params[method]])
        df_means.ix[method] = np.r_[np.nanmean(mses), np.nanmean(betas, axis=0)]
        df_nans.ix[method] = np.r_[np.sum(np.isnan(mses)), 
                                   np.sum(np.isnan(betas), axis=0)]
       
    plt.figure()
    df_means['mse'].plot()
    plt.show()
    plt.figure()
    df_means.ix[:, 1:3].plot()
    plt.show()
    plt.figure()
    df_means.ix[:, 3:].plot()
    plt.show()
                
    plt.figure()
    df_nans.plot()
    plt.show()    
        
        
