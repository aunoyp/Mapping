# -*- coding: utf-8 -*-

def get_xy(neuron):
    x = np.unique(neuron.df['CUE_X'])
    x = x[np.isnan(x)==False]
    y = np.unique(neuron.df['CUE_Y'])
    y = y[np.isnan(y)==False]
    return x, y
             
def get_ylim(neuron):
    ymin = int(np.floor(np.nanmin(neuron.fr_space)))
    ymax = int(np.ceil(np.nanmax(neuron.fr_space)))
    return (ymin, ymax)
    
def get_xy_rew_inds(x, y, irew):
    is_x = np.array(abs(neuron.df['CUE_X'] - x) < 0.01)
    is_y = np.array(abs(neuron.df['CUE_Y'] - y) < 0.01)
    is_rew = np.array(neuron.df['rew'] == irew)
    return np.logical_and(np.logical_and(is_x, is_y), is_rew)
    
def get_fr_by_location(neuron):
    x, y = get_xy(neuron)
    neuron.fr_space = np.empty((2, len(neuron.tStart_smooth), len(x), len(y)))
    for ix in range(len(x)):
        for iy in range(len(y)):
            for irew in range(2):
                inds = get_xy_rew_inds(x[ix], y[iy], irew)           
                neuron.fr_space[irew, :, ix, iy] = np.nanmean(neuron.fr_smooth[inds,:], 0)
                                    
def psth_map(neuron):
    if neuron.fr_smooth == None:
            neuron.smooth_firing_rates(10, 1)                            
    fig, ax = plt.subplots(nrows=len(y), ncols=len(x))
   
    ylim = get_ylim(neuron)
    for xi in range(len(x)):
        for yi in range(len(y)):
            #plot
            plt.sca(ax[len(y) - yi - 1, xi])  
            plt.plot(tMean, fr_time[iCell, xi, yi, 0], color='r')
            plt.plot(tMean, fr_time[iCell, xi, yi, 1], color='b')            
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
            plt.xlim(tFrame)
            plt.ylim(ylim)            
            plt.box()      
    fig.text(0.5, 0.99, 'Cell %d' %(iCell), 
             ha='center', va='center')
    fig.text(0.5, 0.01, 'Time relative to cue onset (s)', 
             ha='center', va='center')
    fig.text(0.01, 0.5, 'Firing rate (sp/s)', 
             ha='center', va='center', rotation='vertical')
    fig.tight_layout()
    title = 'cell %d, psth' %(iCell)        
    plt.savefig(title[0:title.find('\n')] + '.eps', bbox_inches='tight')
    plt.show()