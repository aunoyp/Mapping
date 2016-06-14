# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle


class Population(object):
    ''' Population object:
    PURPOSE: To prepare for population-level analyses by running through all
    neurons, fitting gaussian functions, and save these coefficients (this can
    take a while, especitally when using main iterations for basinhopping
    algorithm)
    FUNCTIONALITY: Fits the functions, and saves these results - to be
    utilized by Classifier object
    '''

    def __init__(self):
        self.io = LoadData()
        self.ncells = 0
        for file in self.io.experiment.files:
            self.ncells += len(self.io.experiment.files[file])
        self.betas = np.empty((1,))
        self.mse = np.empty((self.ncells,))

    def fit_gaussians(self, niter=100):
        icell = 0
        for file in self.io.experiment.files:
            for cell in self.io.experiment.files[file]:
                # load neuron object
                print('Loading neuron', file, cell)
                neuron = self.io.load_neuron(file, cell)
                # X/Y information for stimuli in mapping experiment
                neuron.get_xy()
                neuron.get_xy_grid()
                # Computer firing rates for each location in experiment
                neuron.get_frmean_by_loc((.1,.5))
                # Use firing rates to determine initial guess paramaters for 2d
                # gaussian fit
                neuron.get_initial_params()
                # Fit guassian function with 'basinhopping' algorithm in attempt
                # find a global minimum in this paramater space
                results = neuron.fit_gaussian(niter=niter)
                # record results
                if len(self.betas) == 1:
                    self.betas = np.empty((self.ncells, len(results.x)))
                self.betas[icell,:] = results.x
                self.mse[icell] = results.fun
                icell += 1

    def save_params(self):
        ''' Create a pickle of the neuron object '''
        directory = os.path.join(os.getenv('HOME'), 'GitHub/MappingData/')
        fname = 'populations_params'
        print('saving:', directory + fname)
        with open(directory + fname + '.pickle', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


class PopParams(object):

    def __init__(self):
        self.directory = os.path.join(os.getenv('HOME'), 'GitHub/MappingData/')
        fname = 'model_testng'
        print('saving:', self.directory + fname)
        with open(self.directory + fname + '.pickle', 'rb') as f:
            params = pickle.load(f)

        self.param_labels = ['ux', 'uy', 'stdx', 'stdy',
                             'minfr_nr', 'maxfr_nr', 'minfr_rw', 'maxfr_rw']
        self.method = 'Nelder-Mead'
        self.df = pd.DataFrame([x[1] for x in params[self.method]],
                               columns=self.param_labels)

    def xy_plot_info(self):
        ux = self.df['ux']
        uy = self.df['uy']
        comp = (self.df['minfr_rw'] - self.df['minfr_nr']) / \
               (self.df['minfr_rw'] + self.df['minfr_nr'])
        thresh = 0.6

        pos_mean = (ux[comp > thresh].mean(), uy[comp > thresh].mean())
        pos_sem = (ux[comp > thresh].sem(), uy[comp > thresh].sem())
        neg_mean = (ux[comp < -thresh].mean(), uy[comp < -thresh].mean())
        neg_sem = (ux[comp < -thresh].sem(), uy[comp < -thresh].sem())

        colors = [(0,0,1) if x > thresh
                  else (1,0,0) if x < -thresh
                  else (0.3, 0.3, 0.3) for x in comp]
        size = [30 if abs(x) > thresh
                else 10 for x in comp]
        return ux, uy, pos_mean, pos_sem, neg_mean, neg_sem, colors, size

    def xy_scatter(self):
        ux, uy, pos_mean, pos_sem, neg_mean, neg_sem, colors, size = \
            self.xy_plot_info()

        plt.figure()
        plt.scatter(ux, uy,
                    c=colors, s=size, linewidths=0, alpha=0.5)
        plt.show()

        plt.figure()
        plt.scatter(pos_mean[0], pos_mean[1], c='b', s=30)
        plt.errorbar(pos_mean[0], pos_mean[1], pos_sem[1], pos_sem[0], c='b')
        plt.scatter(neg_mean[0], neg_mean[1], c='r', s=30)
        plt.errorbar(neg_mean[0], neg_mean[1], neg_sem[1], neg_sem[0], c='r')
        plt.show()

    def xy_scatter_bokeh(self):
        from bokeh.plotting import output_server, figure, scatter, show
        ux, uy, pos_mean, pos_sem, neg_mean, neg_sem, colors, size = \
            self.xy_plot_info()
        output_server("scatter.html")
        figure(tools="pan,wheel_zoom,box_zoom,reset,previewsave,select")

        scatter(ux, uy, color="#FF00FF", nonselection_fill_color="#FFFF00", nonselection_fill_alpha=1)
        scatter(ux, uy, color="red")
        scatter(ux, uy, marker="square", color="green")
        scatter(ux, uy, marker="square", color="blue", name="scatter_example")

        show()


if __name__ == '__main__':

    # p = Population()
    # p.fit_gaussians(niter=100)
    # p.save_params()

    pop = PopParams()
    pop.xy_scatter()
