import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc

rc('font', size='9')
mpl.rcParams['svg.fonttype'] = 'none'
sns.mpl.rcParams['svg.fonttype'] = 'none'
sns.set()

dataroot = '/triton/becs/scratch/braindata/shared/TouchHyperScan/hyperclass_test/'

#filename = '/triton/becs/scratch/braindata/shared/TouchHyperScan/hyperclass_test/within_confmat.mat'
filename = '/triton/becs/scratch/braindata/shared/TouchHyperScan/hyperclass_test/hyper_confmat.mat'
data = h5py.File(filename)
mconfmat = np.array(data.get('mean_confmat'))

cmap=plt.cm.Blues
#cmap=plt.cm.jet

# Draw a heatmap with the numeric values in each cell
masknames = ['CunealCortex','IntracalcarineCortex','OccipitalPole',
'PrecuneousCortex','auditory_cortex','combined_loc_mask','FrontalPole','ACC',
'aINS','Amy','COMBINED','INS','ParaCC','pINS','S1','S2','Thalamus']
titles = ['CunealCortex','IntracalcarineCortex','OccipitalPole',
'PrecuneousCortex','Auditory Cortex','Combined Localizer','Frontal Pole','ACC',
'aINS','Amygdala','Combined Atlas','Combined Insula','Paracingulate Cortex',
'pINS','SI','SII','Thalamus']

for i in range(mconfmat.shape[-1]):
    confmat = mconfmat[:,:,i]*100

    fig, ax = plt.subplots()
        
    ax = sns.heatmap(confmat, annot=True, cmap=cmap)
    #ax = sns.heatmap(confmat, annot=True, fmt='.2f', cmap=cmap)
    #ax = sns.heatmap(confmat, cmap=cmap)
    
    mesh=ax.collections[0]
    mesh.set_clim(vmin=0,vmax=100)
    
    cbar_ax = fig.axes[-1]
    cbar_solids = cbar_ax.collections[0]
    #cbar_solids.set_clim(vmin=0,vmax=100)
    cbar_solids.set_edgecolor("face")
    
    ax.set_title(titles[i]);
    
    ax.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off',  # labels along the bottom edge are off
        labelleft='off',
        right='off')
    
    #fig.savefig('{0}/within/{1}/confusion_matrix.svg'.format(dataroot,masknames[i]), format='svg')
    fig.savefig('{0}/hyper/{1}/confusion_matrix.svg'.format(dataroot,masknames[i]), format='svg')
    
    
    
    
