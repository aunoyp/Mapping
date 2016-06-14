# -*- coding: utf-8 -*-

from mapping.neuron import Neuron
import numpy as np
import os
import pandas as pd
import pickle


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
        if type(cells) == str:
            self.add_neuron(f, fname, cells)
        else:
            for cell in cells:
                self.add_neuron(f, fname, cell)
        self.eyes = []
        self.laser = []
        self.process_analog_data(f)
        self.extract_trial_info(f)
        self.directory = os.path.join(os.getenv('HOME'), 'GitHub/MappingData/')

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
                  f['dat'][()]['data']['laser_dt'][i])
            if sum([abs(x - 0.001) > 10e-6 for x in dt]):
                print('analog dt is not 1000hz')
                ipdb.set_trace()
            # check that the start time of analog data collection is about 0,
            # pad with NaNs if start is slightly after zero
            t0 = (f['dat'][()]['data']['eyes_start_t'][i],
                  f['dat'][()]['data']['laser_start_t'][i])
            nans_to_add = {round(x / 1e-3) for x in t0}
            if len(nans_to_add) == 1:
                nans_to_add = nans_to_add.pop()
            else:
                print('different t0 for each analog data type')
                ipdb.set_trace()
            if nans_to_add > 5:
                print('analog start time is > 5 ms')
                ipdb.set_trace()
            # these are lists (for every trial)
            # of numpy arrays (for every data point)
            self.eyes.append(np.vstack((np.full((nans_to_add,2), np.nan),
                                        np.array(f['dat'][()]['data']['eyes'][i],
                                        dtype=float))))
            self.laser.append(np.hstack((np.full((nans_to_add,), np.nan),
                                        np.array(f['dat'][()]['data']['laser'][i],
                                        dtype=float))))

    def get_data_labels(self):
        '''Data fields that we want to keep from the loaded .mat file.
        Specify desired data type for each field as the default is 'object'
        '''
        return [('t_FP_ON', 'float'), ('t_FIX_ACH', 'float'),
                ('t_CUE_ON', 'float'), ('t_CUE_OFF', 'float'),
                ('t_TARG_ON', 'float'), ('t_TARG_OFF', 'float'),
                ('t_FIX_OUT', 'float'), ('t_TARG_ACH', 'float'),
                ('t_SUCCESS', 'float'), ('t_REWARD_ON', 'float'),
                ('t_REWARD_OFF', 'float'), ('t_OUTCOME_OFF', 'float'),
                ('t_TRIAL_END', 'float'), ('t_NO_FIX', 'float'),
                ('t_BREAK_FIX', 'float'), ('t_BREAK_CUE', 'float'),
                ('t_BREAK_TARG', 'float'), ('t_NO_SACCADE', 'float'),
                ('t_MISS_TARG', 'float'), ('t_NO_HOLD_TARG', 'float'),
                ('t_FAIL', 'float'),
                ('CUE_X', 'float'), ('CUE_Y', 'float'),
                ('CUE_TYPE', 'float'),
                ('REPEAT', 'bool')]

    def extract_trial_info(self, f):
        '''Process trial information from the .mat structure. First
        pass this information into a python dict and then transform into a
        pandas dataframe.

        # DICT with all trial information
        # all information is the same (one per trial) EXCEPT for 'spkdata'
        # which can be any number of elements per trial
        # (these are all CAST as OBJECTS - need to specify int or float)

        '''
        data = {}
        labels = self.get_data_labels()
        for label in labels:
            if label[0] == 't_REWARD_ON':
                # 't_REWARD_ON' may include mutliple values - include the 1st
                data[label[0]] = np.empty((self.ntrials,), dtype=label[1])
                for i in range(self.ntrials):
                    times = f['dat'][()]['data'][label[0]][i]
                    if type(times) == float:
                        data[label[0]][i] = times
                    else:
                        data[label[0]][i] = times[0]
            else:
                data[label[0]] = np.array(
                    f['dat'][()]['data'][label[0]], dtype=label[1])

        # add additional fields
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

    def save_session(self):
        '''Save session object'''
        print('saving:', self.filename)
        with open(self.directory + self.filename + '.pickle', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)