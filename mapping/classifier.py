# -*- coding: utf-8 -*-

from itertools import combinations
from mapping.experiment import Experiment, LoadData, get_file_info
from mapping.population import Population
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import pickle
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans

class Classifier(object):
    ''' Classifier object:
    PURPOSE: To do population-level analysis on the
    coefficients of the Gaussian-fits for each individual neuron.
    FUNCTIONALITY: This loads a Population object (with stored, fitted paramters)
    and had methods (mainly from scikit-learn) for analyzing this data.
    '''

    def __init__(self):

        # load population data
        directory = os.path.join(os.getenv('HOME'), 'GitHub/MappingData/')
        fname = 'populations_params'
        with open(directory + fname + '.pickle', 'rb') as f:
            self.pop = pickle.load(f)

        finfo = get_file_info()
        files = finfo['filenames'][finfo['file_ind']]
        self.monkey = np.array([0 if file[0]=='t' else 1 for file in files])
        if self.monkey.shape[0] != self.pop.betas.shape[0]:
            ipdb.set_trace()

        # experiment information - CHANGE TO LOAD FROM ELSEWHERE IN CASE
        # OF FUTURE CHANGES
        self.x = np.array([-12.8, -6.4, 0, 6.4, 12.8])
        self.y = np.array([-12.8, -6.4, 0, 6.4, 12.8])
        self.param_labels = ['ux', 'uy', 'stdx', 'stdy',
                             'minfr_nr', 'maxfr_nr', 'minfr_rw', 'maxfr_rw']

        # plot parameters
        self.xy_max = 20
        self.plot_density = 10 #points per degree
        self.my_colors = ['c', 'r', 'k', 'b', 'm', 'g']

    def get_betas_dict(self, betas):
        return {self.param_labels[i]: betas[:,i]
                for i in range(len(self.param_labels))}

    def get_good_betas(self, monkey=None):
        # NEED TO CHANGE THIS TO SOME GOOD FIT CRITERION
        if monkey == None:
            neurons = np.where(self.pop.mse > 1e-10)[0]
        elif monkey==0 or monkey==1:
            neurons = np.where((self.pop.mse > 1e-10) & (self.monkey==monkey))[0]
        return self.pop.betas[neurons,:]

    def pca(self, n_components=2, plot_it=False):
        X = self.get_good_betas()
        pca = PCA(n_components=n_components)
        X1 = pca.fit_transform(X)
        if plot_it==True:
            plt.figure()
            plt.scatter(X1[:,0], X1[:,1])
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('n = %d' %(X.shape[0]))
            plt.show()
        return X1

    def kmeans(self, X, X1, n_clusters=2):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
        # clustering on dim reduced data
        plt.figure()
        colors = self.get_colors(kmeans.labels_)
        plt.scatter(X1[:,0], X1[:,1], c=colors)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()
        return kmeans.labels_

    def kmeans_exploratory(self, X, labels):
        ''' exporatory analyses comparing proporties of the groups defined
        by a kmeans cluster

        X: betas from 2d gaussian fit
        labels: grouping by kmeans

        self.param_labels = ['ux', 'uy', 'stdx', 'stdy',
                             'minfr_nr', 'maxfr_nr', 'minfr_rw', 'maxfr_rw']
        '''

        colors = self.get_colors(labels)
        for i,j in combinations(range(X.shape[1]), 2):
            if i > 3 and j > 3:
                plt.figure()
                plt.scatter(X[:,i], X[:,j], c=colors)
                plt.xlabel(self.param_labels[i])
                plt.ylabel(self.param_labels[j])
                plt.show()


    def kmeans_targeted(self, X, labels):
        ''' specific analyses comparing proporties of the groups defined
        by a kmeans cluster

        X: betas from 2d gaussian fit
        labels: grouping by kmeans
        '''
        colors = self.get_colors(labels)
        d = self.get_betas_dict(X)
        d_labels = self.get_colors_dict(labels)
        plt.figure()
        x = d['maxfr_rw'] - d['minfr_rw']
        y = d['maxfr_nr'] - d['minfr_nr']
        plt.scatter(x, y, c=colors)
        plt.xlabel('maxfr_rw - minfr_rw')
        plt.ylabel('maxfr_nr - minfr_nr')
        plt.title(d_labels)
        plt.show()

    def get_colors(self, labels):
        return [self.my_colors[x] for x in labels]

    def get_colors_dict(self, labels):
        d = {}
        for i in np.unique(labels):
            d[i] = self.my_colors[i]
        return d

    def heatmaps(self, X, labels):
        ''' plot heatmaps for the mean parameters of each cluster
        '''
        self.get_xy_plot_grid()
        for i in np.unique(labels):
            Xgroup = X[labels==i, :].mean(axis=0)
            g_nr = self.get_gauss(0, Xgroup)
            g_rw = self.get_gauss(1, Xgroup)
            self.plot_gaussian([g_nr, g_rw], ['No reward', 'Reward'],
                               'Kmeans group ' + str(i), Xgroup)

    def plot_gaussian(self, g, cond_labels, title, betas):

        '''plot gaussian colormap
           g is a MxNxK array where K is the number of conditions
           z is actual firing rates
        '''
        fig, ax = plt.subplots(nrows=1, ncols=np.size(g,0))
        plt.suptitle(title)
        for k in range(np.size(g,0)):
            # colormap
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
            # plot the center
            x_scatter = (betas[0] + self.xy_max) * self.plot_density
            y_scatter = (betas[1] + self.xy_max) * self.plot_density
            plt.scatter(x_scatter, y_scatter, marker='o', c='k')
            # format
            divider = make_axes_locatable(ax[k])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.tick_params(labelsize=8)
            plt.title(cond_labels[k])
        fig.tight_layout()
        plt.show()

    def get_gauss(self, is_rew, betas0):
        ''' g = min(4) + max-min(5) *
            exp( - ((x-ux(0))/stdx(2))**2 - ((y-uy(1))/stdy(3))**2)
        '''
        if is_rew:
            k = 2
        else:
            k = 0
        x_part = ((self.x_plot_grid - betas0[0]) / betas0[2]) ** 2
        y_part = ((self.y_plot_grid - betas0[1]) / betas0[3]) ** 2
        g = betas0[k+4] + betas0[k+5] * np.exp(-x_part/2 - y_part/2)
        return g

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


if __name__ == '__main__':
    # create clasifier object
    c = Classifier()
    X = c.get_good_betas(monkey=None)
    X1 = c.pca(plot_it=False)

    # classify neurons into 'n_clusters' groups
    labels = c.kmeans(X, X1, n_clusters=3)
    c.kmeans_exploratory(X, labels)
    c.kmeans_targeted(X, labels)
    c.heatmaps(X, labels)