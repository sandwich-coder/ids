from scipy import integrate
from scipy import stats
from scipy.spatial.distance import pdist, cdist
import matplotlib as mpl
from matplotlib import pyplot as pp
mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.titlesize'] = 'xx-small'
mpl.rcParams['axes.labelsize'] = 'xx-small'
mpl.rcParams['xtick.labelsize'] = 'xx-small'
mpl.rcParams['ytick.labelsize'] = 'xx-small'
mpl.rcParams['legend.fontsize'] = 'xx-small'
mpl.rcParams['lines.markersize'] = 1
mpl.rcParams['lines.linewidth'] = 0.5
import torch
from torch import optim, nn
