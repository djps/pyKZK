""" 

converter.py

David Sinden 2012

This file converts the axisymmetric data into 3D Cartesian data. In this
case the axial interpolation is done by the skip variable and the radial
interpolation is done by simply changing the intervals at which data is 
read.

"""

import numpy as np

import os, time, datetime

import peakdetect, intersections

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from matplotlib.font_manager import FontProperties

from scipy.io import loadmat, savemat

# clear screen
os.system('cls' if os.name == 'nt' else 'clear')

# load canonical data set
data = loadmat("../../../../Code/least squares bioheat/Madden/Axi/Computed_Data/thermal_5s_-11dbm_1.mat")

# define colours for figures
colorscheme=1
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

verbose = 1

Z = data['Z']
d = data['d']
R = data['R']
a = data['a']
p0 = data['p0']
p5x = data['p5x']
p5r = data['p5r']
peak = data['peak']
trough = data['trough']
z = data['z']
r = data['r']
z_bht = data['z_bht']
r_bht = data['r_bht']
Dmat = data['Dmat']
Pneg = data['Pneg']
Ppos = data['Ppos']

# Rescale quantities according to non-dimensionalisation scheme
Z = d*Z
R = a*R
p5x = 1e-6*p0*p5x
p5r = 1e-6*p0*p5r
peak = 1e-6*p0*peak
trough = 1e-6*p0*trough
K,N = np.shape(p5x)

# get minima of trough 
max_peaks = peakdetect.peakdetect( np.abs(trough), delta=1 )

print '\n', max_peaks, len(z), len(trough)

x_peak, peak_value = np.transpose(max_peaks)

print '\n', x_peak, z[0,int(x_peak[0])], 
print '\n', peak_value, np.abs(trough[int(x_peak[0]),0])

# minpeakdistance is somewhat arbitary, but it enables the secondary peak to be identified.
#dummy,ii = np.sort(pks)

#print pks, loc

#axial_secondary_value = trough[ int(loc[ int(ii[-2]) ] ) ]
#axial_secondary_location = z[int(loc[int(ii[-2])])]

#radial = np.sum( np.transpose(p5r[0:-1,:]) )
#pks,loc = peakdetect.peakdetect( radial, delta=3)
#dummy,ii = np.sort(pks)
#radial_secondary_value = np.sum( p5r[loc[ii[-1]],:] )
#radial_secondary_location = r[loc[ii[-1]]]

#-------------------------------------------------------------------------#
#threshold = -1.05*np.maximum( (radial_secondary_value, axial_secondary_value) )
#-------------------------------------------------------------------------#

threshold = -1.0

# get intersections of peak negative pressure threshold value and peak 
# negative pressure field

# axial
#[x0, y0, iout, jout] 
#np.column_stack( (roots, p1(roots), p2(roots) ) ) = intersections.intersections( z, threshold*np.ones((np.shape(z)[1],1)), z, trough )
roots, threshold_roots, trough_roots = intersections.intersections( np.squeeze(z), np.squeeze(np.transpose(threshold*np.ones((np.shape(z)[1],1)))), \
      np.squeeze(z), np.squeeze(np.transpose(trough)) )

font0 = FontProperties()
font1 = font0.copy()
font1.set_size('large')

fig1, ax = plt.subplots()
fig1.hold(True)

# plt.set_size(12) <-- false!
# ax.set_size(12) <-- false!

#plt.fill( [z[iout[0]:iout[1]]'; z[iout[1]:-1:iout[0]]'; z[iout[0]] ], [peak[iout[0]:iout[1]]; trough[iout[1]:-1:iout[0]]; peak[iout[0]] ], ICRbrightred, 'EdgeColor', 'r' )
#plt.plot( z[iout[0]], trough[iout[0]], 'o', 'MarkerEdgeColor', ICRblue, 'MarkerFaceColor', ICRblue, 'MarkerSize', 6)
#plt.plot( z[iout[1]], trough[iout[1]], 'o', 'MarkerEdgeColor', ICRblue, 'MarkerFaceColor', ICRblue, 'MarkerSize', 6)

#plt.plot( axial_secondary_location, axial_secondary_value, '*', 'MarkerEdgeColor', ICRolive, 'MarkerFaceColor', ICRblue, 'MarkerSize', 6)

ax.plot( np.squeeze(z), np.squeeze(np.transpose(peak)), linewidth=2, linestyle='-', color=ICRcolors['ICRred'] )
ax.plot( np.squeeze(z), np.squeeze(np.transpose(trough)), linewidth=2, linestyle='-', color=ICRcolors['ICRblue'])

ax.plot( np.squeeze(z), threshold*np.squeeze( np.ones( (np.shape(z)[1],1) ) ), linewidth=1, linestyle='--', color=ICRcolors['ICRgreen'] )

#ylim = get[gca,'YLim']; 
#axis[[0,Z,ylim[1],ylim[2]]]

#plt.plot( [z[iout[0]],z[iout[0]]], [ylim[0],ylim[1]], 'LineWidth', 1, 'LineStyle', '--', 'Color', ICRblue)
#plt.plot( [z[iout[1]],z[iout[1]]], [ylim[0],ylim[1]], 'LineWidth', 1, 'LineStyle', '--', 'Color', ICRblue)

#plt.xlim(0, Z)
#yticks = np.arange(-50, 30, 10)
xticks = np.arange(0,Z,10)

ax.set_xlabel('z [cm]', fontsize=14, color='red')
ax.set_ylabel('P [MPa]', fontsize=14, color='red')
ax.set_title('Peak Pressures')

ax.set_xticks(xticks,minor=True)

plt.grid(True)

plt.show()



