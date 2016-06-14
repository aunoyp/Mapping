# -*- coding: utf-8 -*-

from collections import OrderedDict
from mapping.session import Session
from mapping.neuron import Neuron
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import pickle
import scipy as sp
import scipy.io



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

        self.directory = os.path.join(os.getenv('HOME'), 'GitHub/MappingData/')

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
        self.directory = os.path.join(os.getenv('HOME'), 'GitHub/MappingData/')
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


def demo(one_example=True):

    io = LoadData()
    if one_example == True:
        demo_neurons = {'tn_map_123014': ['elec11U']}
    else:
        demo_neurons = {'tn_map_120914': ['elec10a', 'elec21a'],
                        'tn_map_123014': ['elec11U', 'elec9U'],
                        'tn_map_122914': ['elec7U'],
                        'tn_map_121614': ['elec17U', 'elec3b'],
                        'tn_map_121214': ['elec13U'],
                        'tn_map_121114': ['elec11U']}
    for file in demo_neurons:
        for cell in demo_neurons[file]:

            # load neuron object
            print('Loading neuron', file, cell)
            neuron = io.load_neuron(file, cell)

            # X/Y information for stimuli in mapping experiment
            neuron.get_xy()
            neuron.get_xy_grid()
            neuron.get_xy_plot_grid()

            # Firing sorted by reward condition (ignoring spatial location)
            neuron.smooth_firing_rates()
            neuron.psth_rew()

            # Firing rates for each location in experiment
            neuron.get_frmean_by_loc((.1,.5))
            neuron.get_fr_by_loc()
            neuron.psth_map()

            # Use firing rates to determine initial guess paramaters for 2d
            # gaussian fit
            neuron.get_initial_params()

            # Fit guassian function with 'basinhopping' algorithm in attempt
            # find a global minimum in this paramater space
            neuron.fit_gaussian()

            # Get high-res guassian for plotting based on the fitted parameters
            g_nr = neuron.get_gauss(neuron.x_plot_grid, neuron.y_plot_grid, 0,
                                    neuron.betas)
            g_rw = neuron.get_gauss(neuron.x_plot_grid, neuron.y_plot_grid, 1,
                                    neuron.betas)
            neuron.plot_gaussian([g_nr, g_rw], neuron.frmean_space)


def get_file_info():
    # initialize Experiment, and load information
    finfo = sp.io.loadmat(
        os.path.join(os.getenv('HOME'), 'GitHub/MappingData/src/map_cell_list.mat'),
        squeeze_me=True
    )
    # changed to zero based inds
    finfo['cell_ind'] -= 1
    finfo['file_ind'] -= 1
    return finfo

def create_all(overwrite=True):
    ''' Create all Sessions & Neuron objects '''
    finfo = get_file_info()
    directory = os.path.join(os.getenv('HOME'), 'GitHub/MappingData/src/')
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
    directory = os.path.join(os.getenv('HOME'), 'GitHub/MappingData/src/')
    for iFile, fname in enumerate(finfo['filenames']):
        f = sp.io.loadmat(directory + fname + '.nex.mat', squeeze_me=True)
        print('Loaded file', fname)
        session = Session(f, fname, [])
        session.save_session()

def load_neurons_and_plot(start_ind=0):
    ''' load all Neuron objects and plot them '''
    io = LoadData()
    fig_dir = os.path.join(os.getenv('HOME'), 'GitHub/Mapping/figs/')
    for i, file in enumerate(io.experiment.files):
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

                    neuron.psth_map(pdf)
                    neuron.psth_rew(pdf)

                    #min_pos, max_pos = neuron.define_hot_spot()
                    #neuron.plot_hot_spot(min_pos, max_pos)

                    # Use firing rates to determine initial guess paramaters for 2d
                    # gaussian fit
                    neuron.get_initial_params()

                    # Fit guassian function with 'basinhopping' algorithm in attempt
                    # find a global minimum in this paramater space
                    neuron.fit_gaussian(niter=1)

                    # Get high-res guassian for plotting based on the fitted parameters
                    g_nr = neuron.get_gauss(neuron.x_plot_grid, neuron.y_plot_grid, 0,
                                            neuron.betas)
                    g_rw = neuron.get_gauss(neuron.x_plot_grid, neuron.y_plot_grid, 1,
                                            neuron.betas)
                    neuron.plot_gaussian([g_nr, g_rw], neuron.frmean_space, pdf)


if __name__ == '__main__':

    #create_all(overwrite=True)

    #load_neurons_and_plot(start_ind=0)
    #create_all(overwrite=True)
    #overwrite_sessions()
    demo()
