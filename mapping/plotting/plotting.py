#!/usr/bin/env python
'''
Commonly used functions for plotting with matplotlib.
'''

from __future__ import print_function, absolute_import, division
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.font_manager import fontManager, FontProperties
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def customize_mpl(fontsize=20, labelfontsize=None, legendfontsize=None, grid=False, bgcolor_black=False):
    '''Customizes plot colors and other visual entities'''
    if not labelfontsize:
        labelfontsize = fontsize
    mpl.rcParams.update({'figure.facecolor': 'w'})
    mpl.rcParams.update({'axes.edgecolor': '#999999'})
    mpl.rcParams.update({'axes.labelcolor': '#333333'})
    mpl.rcParams.update({'text.color': '#333333'})
    mpl.rcParams.update({'xtick.color': '#333333'})
    mpl.rcParams.update({'ytick.color': '#333333'})
    mpl.rcParams.update({'font.size': fontsize})
    mpl.rcParams.update({'legend.fontsize': 14})
    mpl.rcParams.update({'axes.labelsize': labelfontsize})
    if legendfontsize: mpl.rcParams.update({'legend.fontsize':legendfontsize})
    if grid: mpl.rcParams.update({'axes.grid': True})
    if bgcolor_black:
        mpl.rcParams.update({'text.color': 'w'})
        mpl.rcParams.update({'axes.facecolor': 'k'})
        mpl.rcParams.update({'axes.edgecolor': 'w'})
        mpl.rcParams.update({'axes.labelcolor': 'w'})
        mpl.rcParams.update({'xtick.color': 'w'})
        mpl.rcParams.update({'ytick.color': 'w'})
        mpl.rcParams.update({'grid.color': 'w'})
        mpl.rcParams.update({'figure.facecolor': 'k'})
        mpl.rcParams.update({'figure.edgecolor': 'k'})

def setup_ax_arr(fig, numV, numH, ylim, sharey=True, hide_xlabels=True, hide_ylabels=True):
    # hide_label option only applies to subplots on the left most and bottom, other labels will always be hidden.
    ax_arr = []
    for i in np.arange(numV*numH):
        if i==0:
            ax_arr.append(fig.add_subplot(numV,numH,i+1))
        else:
            if sharey:
                ax_arr.append(fig.add_subplot(numV,numH,i+1, sharex=ax_arr[0], sharey=ax_arr[0]))
            else:
                ax_arr.append(fig.add_subplot(numV,numH,i+1, sharex=ax_arr[0]))
        ax_arr[i].set_ylim(ylim[0],ylim[1])
        if (not hide_xlabels) and ( i > (numV-1)*numH):
            pass
        else:
            for label in ax_arr[i].get_xticklabels():
                label.set_visible(False)
        if (not hide_ylabels) and ( (i-1) % numH == 0):
            pass
        else:
            for label in ax_arr[i].get_yticklabels():
                label.set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    return ax_arr

def setup_axes(ax, title=None, ylabel=None, ylim=None, aligny=False, xlim=None, xlabel=None, ygrid=False, xgrid=False, num_major_xticks=None, num_major_yticks=5, xtick_labels=False, ytick_labels=True, vlines=[], hlines=[], xlog=False, ylog=False, topright=False):
    if title: ax.set_title(title)
    if ylim: ax.set_ylim(ylim[0], ylim[1])
    if ylabel: ax.set_ylabel(ylabel, multialignment='center')#, ha='center')
    if aligny: ax.yaxis.set_label_coords(-.3, 0.5)
    if xlim: ax.set_xlim(xlim[0], xlim[1])
    if xlabel: ax.set_xlabel(xlabel)
    if ygrid: ax.yaxis.grid(b=True)
    if xgrid: ax.xaxis.grid(b=True)
    if num_major_xticks: ax.xaxis.set_major_locator(MaxNLocator(num_major_xticks))
    if num_major_yticks: ax.yaxis.set_major_locator(MaxNLocator(num_major_yticks))
    if not xtick_labels:
        for label in ax.get_xticklabels():
            label.set_visible(False)
    if not ytick_labels:
        for label in ax.get_yticklabels():
            label.set_visible(False)
    linecolor = 'k'
    for vline in vlines:
        ax.axvline(vline, ls='--', color=linecolor, alpha=0.8)
    for hline in hlines:
        ax.axhline(hline, ls='-.', color=linecolor, alpha=0.8)
    if not topright:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

class colors:
    reds = ['#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026']
    blues = ['#d0d1e6', '#a6bddb', '#74a9cf', '#3690c0', '#0570b0', '#045a8d', '#023858']
    blue_yellows = ['#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#081d58']
    greens = ['#ccece6', '#99d8c9', '#66c2a4', '#41ae76', '#238b45', '#006d2c', '#00441b']
    greys = ['#d9d9d9', '#bdbdbd', '#969696', '#737373', '#525252', '#252525', '#111111']
    grays = ['#d9d9d9', '#bdbdbd', '#969696', '#737373', '#525252', '#252525', '#111111']
