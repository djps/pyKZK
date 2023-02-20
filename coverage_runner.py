import os, time, datetime, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from math import pi, log

from termcolor import colored, cprint

from scipy.io import loadmat, savemat
from scipy.sparse import spdiags, eye, bmat

from scipy.sparse.linalg import spsolve, factorized, splu
from scipy.linalg import solve, lu

from scipy.io import loadmat, savemat

from scipy.sparse import SparseEfficiencyWarning

# suppress warnings
warnings.simplefilter("ignore", SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import WAKZK_planar, WAKZK, WAKZK_Gaussian
import SourceFilterH, SynthAxScan, SynthRadScan

willplot=False

Grid, Layer, Q = WAKZK_Gaussian.WAKZK_Gaussian(willplot)
