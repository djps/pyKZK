#--------------------------------------------------------------------------------------------------
# preamble
  
# load packages
from scipy.optimize import leastsq  
import os, time, datetime, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from math import pi, log

from termcolor import colored, cprint

from scipy.sparse.linalg import spsolve, factorized, splu
from scipy.linalg import solve, lu

from scipy.io import loadmat, savemat

import axisymmetricKZK, equivalent_time, timing, KZK_parameters, BHT_parameters, BHT_operators

from scipy.sparse import SparseEfficiencyWarning

# suppress warnings
warnings.simplefilter("ignore", SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=FutureWarning) 

# experimental data
experimental_filename='20dBm_3_0_0_0_20140516_4s_1c.txt'

# check whether file exists 
if os.path.isfile(experimental_filename):
  # load data
  data = np.genfromtxt(experimental_filename,skip_header=3,usecols=(0,2))
else:
  print " gah : experimental file does not exist "
  exit(2) 
  
# record when channel switching occurs
channelswitch = data[ np.isnan(data).any(axis=1) ]
    
# remove data at instances of channel switching
data = data[ ~np.isnan(data).any(axis=1) ]

# times of channel switching
jump = np.interp( channelswitch[:,0], data[:,0], data[:,1],  )

# automated start-time detection
jtstart = np.argmax( np.gradient(data[:,1]) )

# automated detection of when transducer is switched off
jtoff = np.argmax( data[:,1] ) 

# specify duration of relevant recorded cool-off period
Tend = data[jtoff,0] + 10.0

# index of last relevant time point
jtend = np.argmin( np.abs(data[:,0] - Tend) )

# number of sampling points of experimental data
Ndivisions = 200

# time step for sampled experimental data
Nstep = np.int( np.floor( jtend/Ndivisions ) ) 

# experimental time and temperature curves
t = data[ 0 : jtend-1 : Nstep, 0]
y = data[ 0 : jtend-1 : Nstep, 1]

# get index which specifies time at which transducer was switched on
itstart = np.argmin( np.abs(t - data[jtstart,0] ) )

# get index which specifies last relevant time point
itend = np.argmin( np.abs(t - Tend) )

# sampled experimental data
tdata = t[itstart:itend] - t[itstart]
ydata = y[itstart:itend] - y[itstart]

# index of last relevant time point
jtend = np.argmin( np.abs(channelswitch[:,0] - Tend) )
channelswitch = channelswitch[0:jtend,0]- t[itstart]
jump = jump[0:jtend] - y[itstart]


#print np.int( np.shape(jump)[0] ) - 1 
diff = np.zeros( ( np.shape(jump)[0]-1,) )
time_channel_switch = []
temp_channel_switch = []
for i in np.arange(0, np.int( np.shape(jump)[0] )-1 ):
  diff[i] = jump[i] - jump[i+1]
  if (np.abs(diff[i]) > 0.02):
    print i, jump[i]
    time_channel_switch = np.append(time_channel_switch, channelswitch[i])
    temp_channel_switch = np.append(temp_channel_switch, jump[i])

print time_channel_switch, temp_channel_switch
    
# specify whether colours for figures are defined as rgb or cmyk 
colorscheme=1

# define dictionary of colours
if colorscheme==1:
  ICRcolors = { \
  'ICRgray': (98.0/255.0, 100.0/255.0, 102.0/255.0), \
  'ICRgreen':(202.0/255.0, 222.0/255.0, 2.0/255.0), \
  'ICRred': (166.0/255.0, 25.0/255.0, 48.0/255.0), \
  'ICRpink': (237.0/255.0, 126.0/255.0, 166.0/255.0), \
  'ICRorange': (250.0/255.0, 162.0/255.0, 0.0/255.0), \
  'ICRyellow': (255.0/255.0, 82.0/255.0, 207.0/255.0), \
  'ICRolive': (78.0/255.0, 89.0/255.0, 7.0/255.0), \
  'ICRdamson': (79.0/255.0, 3.0/255.0, 65.0/255.0), \
  'ICRbrightred': (255.0/255.0, 15.0/255.0, 56.0/255.0), \
  'ICRlightgray': (59.0/255.0, 168.0/255.0, 170.0/255.0), \
  'ICRblue': (0.0/255.0, 51.0/255.0, 41.0/255.0)} 
else:
  ICRcolors = { \
  'ICRgray': (0.04, 0.02, 0.00, 0.60), \
  'ICRgreen': (0.09, 0.00, 0.99, 0.13), \
  'ICRred': (0.00, 0.85, 0.71, 0.35), \
  'ICRpink': (0.00, 0.47, 0.30, 0.07), \
  'ICRorange': (0.00, 0.35, 1.00, 0.02), \
  'ICRyellow': (0.00, 0.68, 0.19, 0.00), \
  'ICRolive': (0.24, 0.13, 0.93, 0.60), \
  'ICRdamson': (0.21, 0.97, 0.35, 0.61), \
  'ICRbrightred': (0.00, 0.94, 0.78, 0.00), \
  'ICRlightgray': (0.16, 0.11, 0.10, 0.26), \
  'ICRblue': (1.00, 0.00, 0.20, 0.80) }
  
# render text with TeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# define figure and axis handles
fig1, ax1 = plt.subplots()
# hold on for multiple plots on figure
fig1.hold(True)
# apply grid to figure
plt.grid(True)

# plot experimental temperatures
plot1, = ax1.plot( tdata, ydata, linewidth=2, linestyle='-', color=ICRcolors['ICRred'], label='Experimental' )
ax1.plot( tdata, ydata, marker='o', color=ICRcolors['ICRred'] )
#ax1.scatter( channelswitch, jump, marker='+', color=ICRcolors['ICRblue'] )
# define xticks
xticks1 = np.linspace( np.squeeze(tdata[0]), np.squeeze(tdata[-1]), num=10)
# set minor ticks on both axes
ax1.minorticks_on()
# define xlabel
ax1.set_xlabel(r'$t$ [sec]', fontsize=14, color='black')
# define ylabel
ax1.set_ylabel(r'$\Delta T$ [Degrees]', fontsize=14, color='black')
# set title
ax1.set_title(r'Peak Temperature Rise', fontsize=14, color='black')
# set limits of x-axis
ax1.set_xlim( [np.squeeze(tdata[0]), np.squeeze(tdata[-1])] )
# apply xticks
ax1.set_xticks(xticks1,minor=True)

for i in np.arange(0,np.shape(temp_channel_switch)[0]-1):
  plot4, = ax1.plot( time_channel_switch[i], temp_channel_switch[i], marker='+', color=ICRcolors['ICRblue'] )
  plt.axvline(time_channel_switch[i], color='black')

#plot1.tight_layout()
# set axis for plot
#ax1.legend( [plot1, plot2, plot3], [r'Experimental', r'Least Squares', r'Book Values'] )
# render figures
plt.show()