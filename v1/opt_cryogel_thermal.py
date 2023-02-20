"""

Script: opt_cryogel_thermal.py
Author: David Sinden <david.sinden@icr.ac.uk>
Copyright: (c) 2014, David Sinden / Institute of Cancer Research
License: GPL.
Date: 02/06/2014

This script performs a least squares optimization procedure which matches a set of thermal 
parameters, "p" which, when substituted into an axisymetric bioheat equation, called func, to 
match a temperature curve from a fibre-optic needle hydrophone: "experimental_filename". The 
parameters in p contain the thermal diffusivity and the specific heat capacity : there is no 
perfusion in cryogel.

Care is needed in that the automated detection of the start of the ultrasound exposure is defined 
as the point at which the gradient of the temperature curve is maximal. While this is true 
theoretically for a static and constant rate of heating, channel switching may result in 
spurious results.

This procedure requires a precomputed acoustic field to provide the heating rate 
"acoustic_filename". For universality and cross-platform deployability the heating rate and 
domain size are loaded in matlab .mat format. 

The subroutine leastsq is a wrapper around MINPACK's lmdif and lmder algorithms. The purpose of 
LMDIF is to minimize the sum of the squares of M nonlinear functions in N  variables by a 
modification of the Levenberg-Marquardt algorithm.  The user must provide a subroutine which 
calculates the functions.  The Jacobian is then calculated by a forward-difference approximation.
The purpose of LMDER is to minimize the sum of the squares of M nonlinear functions in N 
variables by a modification of the Levenberg-Marquardt algorithm.  The user must provide a 
subroutine which calculates the functions and the Jacobian.

covx is a Jacobian approximation to the Hessian of the least squares objective function. This 
approximation assumes that the objective function is based on the difference between some 
observed target data (ydata) and a (non-linear) function of the parameters f(xdata, params)

This program is free software; you can redistribute it and/or modify it under the terms of the 
GNU General Public License as published by the Free Software Foundation; either version 2 of the 
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if 
not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 
02110-1301, USA.

"""

#--------------------------------------------------------------------------------------------------
    
def func(p, acoustic_filename, experimental_filename, iterate, drive, efficiency):

  #-------------------------------------------------
  # preamble, warnings and information, clear screen
  #-------------------------------------------------
  
  try:
    import numpy as np
  except ImportError, err0:
    pass 
  # module doesn't exist, deal with it.
    
  import os 
  import time 
  import datetime 
  import warnings

  from math import pi, log
  
  from scipy import transpose
  
  from scipy.sparse import spdiags, eye
    
  from scipy.sparse.linalg import spsolve, factorized, splu
  
  from scipy.linalg import solve, lu

  import matplotlib.pyplot as plt
  import matplotlib.mlab as mlab

  from scipy.io import loadmat, savemat

  from scipy.sparse import SparseEfficiencyWarning
  
  warnings.simplefilter("ignore", SparseEfficiencyWarning)

  # set level of verbose output
  iverbose = 0
  
  # set degree of data loading 
  iload = 0
  
  # specify whether figure is rendered
  ifig = 0
  
  # define whether data is loaded
  manual = 1
  
  # set whether LU decomposition is applied to spead up solution method.
  ilu = 1
   
  # get data to fit against
  if os.path.isfile(experimental_filename):
    data = np.genfromtxt(experimental_filename,skip_header=3,usecols=(0,2))
  else:
    print " gah : experimental file does not exist "
    exit(2)
    
  # remove data at instances of channel switching
  data = data[ ~np.isnan(data).any(axis=1) ]
  
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
  
  ## plot data
  #if (ifig==1):
    ## specify whether colours for figures are defined as rgb or cmyk 
    #colorscheme=1
    ## define dictionary of colours
    #if colorscheme==1:
      #ICRcolors = { \
      #'ICRgray': (98.0/255.0, 100.0/255.0, 102.0/255.0), \
      #'ICRgreen':(202.0/255.0, 222.0/255.0, 2.0/255.0), \
      #'ICRred': (166.0/255.0, 25.0/255.0, 48.0/255.0), \
      #'ICRpink': (237.0/255.0, 126.0/255.0, 166.0/255.0), \
      #'ICRorange': (250.0/255.0, 162.0/255.0, 0.0/255.0), \
      #'ICRyellow': (255.0/255.0, 82.0/255.0, 207.0/255.0), \
      #'ICRolive': (78.0/255.0, 89.0/255.0, 7.0/255.0), \
      #'ICRdamson': (79.0/255.0, 3.0/255.0, 65.0/255.0), \
      #'ICRbrightred': (255.0/255.0, 15.0/255.0, 56.0/255.0), \
      #'ICRlightgray': (59.0/255.0, 168.0/255.0, 170.0/255.0), \
      #'ICRblue': (0.0/255.0, 51.0/255.0, 41.0/255.0)} 
    #else:
      #ICRcolors = { \
      #'ICRgray': (0.04, 0.02, 0.00, 0.60), \
      #'ICRgreen': (0.09, 0.00, 0.99, 0.13), \
      #'ICRred': (0.00, 0.85, 0.71, 0.35), \
      #'ICRpink': (0.00, 0.47, 0.30, 0.07), \
      #'ICRorange': (0.00, 0.35, 1.00, 0.02), \
      #'ICRyellow': (0.00, 0.68, 0.19, 0.00), \
      #'ICRolive': (0.24, 0.13, 0.93, 0.60), \
      #'ICRdamson': (0.21, 0.97, 0.35, 0.61), \
      #'ICRbrightred': (0.00, 0.94, 0.78, 0.00), \
      #'ICRlightgray': (0.16, 0.11, 0.10, 0.26), \
      #'ICRblue': (1.00, 0.00, 0.20, 0.80) }
    ## render text with TeX
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    ## define figure and axis handles
    #fig1, ax1 = plt.subplots()
    ## hold on for multiple plots on figure
    #fig1.hold(True)
    ## plot peak temperature
    #ax1.plot( tdata, ydata, linewidth=2, linestyle='-', color=ICRcolors['ICRred'] )
    ##ax1.plot( t[itstart], y[itstart], marker='*', color=ICRcolors['ICRblue'] )
    #ax1.plot( data[jtstart:jtend,0], data[jtstart:jtend,1], linewidth=1, linestyle='-.', color=ICRcolors['ICRgreen'] )
    #ax1.plot( data[jtstart,0], data[jtstart,1], marker='o', color=ICRcolors['ICRolive'] )
    ## define xticks
    ##xticks = np.arange( t[itstart],t[itend],10)
    ## define xlabel
    #ax1.set_xlabel(r'$t$ [sec]', fontsize=14, color='black')
    ## define ylabel
    #ax1.set_ylabel(r'$T$ [Degrees]', fontsize=14, color='black')
    ## set title
    #ax1.set_title(r'Peak Temperature', fontsize=14, color='black')
    ## apply xticks
    #ax1.set_xticks(xticks,minor=True)
    ## set minor ticks on both axes
    #ax1.minorticks_on()
    ## apply grid to figure
    #plt.grid(True)
    ## render figure
    #plt.show()
  
  # Drive settings
  #-----------------------------

  # start time [seconds]
  tstart = tdata[0]

  # end time [seconds]
  tend = tdata[-1]

  # index at which transducer is switched off
  itoff = np.argmax( ydata ) 
  
  # time at which transducer is switched off [seconds]
  toff = tdata[ itoff ]
  
  # determine proper timestep:
  dt = tdata[1]

  # number of time points
  N = np.shape(tdata)[0]
  
  # duration of exposure
  sequence = np.zeros( (N,1) )
  for i in np.arange(0, itoff+1):
    sequence[i] = 1.0     
  
  # check whether file exists
  if os.path.isfile(acoustic_filename):
    # load data
    data = loadmat(acoustic_filename, variable_names=['z','r','H'] )
    z = data['z']
    r = data['r']
    H = data['H']
    # format data
    z = np.squeeze(z)
    r = np.squeeze(r)  
  else:
    print "No file to load in this instance!"
    exit(1)
  
  # material 1 parameters: WATER
  #-----------------------------
  
  # small-signal sound speed [m/s]
  c01     = 1482.0 
  
  # mass density [kg/m^3]
  rho1    = 1000.0 
  
  # attenuation at 1MHz [dB/m]
  alpha01 = 0.217  
  
  # power of attenuation vs frequency curve
  eta1    = 2
  
  # nonlinearity parameter
  beta1   = 3.5
  
  # absorbtion at 1MHz [dB/m]
  theta01 = 0.217 
  
  # power of absorbtion vs frequency curve
  phi1    = 2    
  
  # heat capacity [J/kg/K]
  Cv1 = 4180.0 
  
  # thermal conductivity [W/m/K]
  k1 = 0.6  
  
  # perfusion rate [kg/m^3/s]
  w1 = 0.0  

  # material 2 parameters: LIVER
  #-----------------------------
  
  # small-signal sound speed [m/s]
  c02      = 1570.0 
  
  # mass density [kg/m^3]
  #rho2    = 1070.0 
  
  # attenuation at 1MHz [dB/m]
  alpha02 = 27.1
  
  # power of attenuation vs frequency curve
  eta2    = 1.223  
  
  # nonlinearity parameter
  beta2   = 4.3525 
  
  # absorbtion at 1MHz [dB/m]
  theta02 = 33  
  
  # power of absorbtion vs frequency curve
  phi2    = 1.1 
  
  ## heat capacity [J/kg/K]
  #Cv2 = 4180.0 
  
  ## thermal conductivity [W/m/K]
  #k2 = 0.6  
  
  # perfusion rate [kg/m^3/s]
  w2 = 0.0  

  # transducer parameters
  #-----------------------------
  
  # outer radius [cm]
  a   = 3.2 
  
  # inner radius [cm]           
  b   = 2.0 
  
  # focusing depth [cm]             
  d   = 6.32 
  
  # frequency [Hz]            
  f   = 1.68e6  

  # computational domain size
  #-----------------------------
  
  # max radius [cm]
  R = a 
  
  # max axial distance [cm]
  Z = 1.5*d 
  
  # propation distance through tissue [cm]
  ztissue = 2.25
  
  # material transition distance [cm]
  z_ = d - ztissue 

  # nondimensionalize grid dimensions:
  R = R/a
  Z = Z/d

  # ambient temperature [degrees C]
  T0 = 37.0
  
  # grid specification
  #-----------------------------
  
  # reduction factor in r-direction
  r_skip = 2
  
  # reduction factor in z-direction
  z_skip = 4          
                 
  # size of J,M based on acoustic models
  J, M = np.size(H,axis=0), np.size(H,axis=1)   
 
  # rescaled radial
  r_bht = r[ 0 : np.round(J/2) : r_skip ]

  # rescaled axial
  z_bht = z[ 0 : M+1 : z_skip ]
  
  # spatial step sizes
  dr = r_bht[1]
  dz = z_bht[1]
                    
  # rescaled heating rate
  H_bht = H[ 0 : np.round(J/2) : r_skip, 0 : M-1 : z_skip ]
  
  # reassign J,M to J_bht, M_bht
  J_bht, M_bht = np.size(H_bht, axis=0), np.size(H_bht, axis=1)
  J_bht, M_bht = np.size(r_bht, axis=0), np.size(z_bht, axis=0)
  JM_bht = J_bht*M_bht

  # index at material interface 
  zz = z_/d/Z
  m_ = np.around(zz*M_bht)
  if (m_>M_bht): 
    m_ = M_bht 

  # construction of discretised operators
  #-----------------------------
    
  # nondimensional diffusivity of material 1 [dimensionless]
  D1 = 1e4*k1/Cv1/rho1

  # nondimensional diffusivity of material 2 [dimensionless]
  D2 = p[0]

  # nondimensional perfusion/density of material 1 [dimensionless]
  P1 = w1/rho1

  # nondimensional perfusion/density of material 2 [dimensionless]
  P2 = 0.0

  # build matrix operator's vector "bands" 
  alpha_plus  = np.squeeze( np.ones( (JM_bht,1), dtype=np.float64 )/dz/dz )
  alpha_minus = np.squeeze( np.ones( (JM_bht,1), dtype=np.float64 )/dz/dz )
  
  bp = np.squeeze( np.zeros( (J_bht,1), dtype=np.float64 ))
  bm = np.squeeze( np.zeros( (J_bht,1), dtype=np.float64 ))
  
  bp[0] = 2.0/dr/dr
  bp[1:J] = (1.0/dr + 0.5/r_bht[1:J] )/dr
  bm[1:J] = (1.0/dr - 0.5/r_bht[1:J] )/dr
  
  beta_plus  = np.squeeze( np.zeros( (JM_bht,1), dtype=np.float64) )
  beta_minus = np.squeeze( np.zeros( (JM_bht,1), dtype=np.float64) )
  
  # jm runs from 0 to JM-1
  for jm in np.arange(0,JM_bht):
    
    if (np.mod(jm+1,M_bht)==0): 
      alpha_plus[jm] = 0.0   
    
    if (np.mod(jm+1,M_bht)==1): 
      alpha_minus[jm] = 0.0
    
    # number of times iterate is a multiple of M
    jn = np.ceil( (jm+0)/M_bht )
    beta_plus[jm]  = bp[jn]
    beta_minus[jm] = bm[jn]
  
  gamma_0 = np.squeeze( -2.0*(1.0/dr/dr + 1.0/dz/dz)*np.ones( (JM_bht,1), dtype=np.float64) )

  # build diffusivity coefficient matrix
  k1minor = D1*np.ones( (m_,1), dtype=np.float64)
  k2minor = D2*np.ones( (M_bht-m_,1), dtype=np.float64)
  k = np.vstack((k1minor,k2minor))
  bigk = k
  
  # j runs from 0 to J-2, i.e. in total there are J-1 items
  for j in np.arange(0,J_bht-1):
    bigk = np.vstack((bigk,k))
  
  K = spdiags(np.squeeze(bigk), 0, JM_bht, JM_bht );

  # build perfusion coefficeint matrix:
  p1 = P1*np.ones( (m_,1), dtype=np.float64 )
  p2 = P2*np.ones( (M_bht-m_,1), dtype=np.float64 )
  pp  = np.vstack((p1,p2))
  bigp = pp
  for j in np.arange(0,J-1):
    bigp = np.vstack((bigp,pp))
  
  P = spdiags(np.squeeze(bigp), 0, JM_bht, JM_bht)
  
  rows = np.array([np.squeeze(beta_plus),np.squeeze(alpha_plus),np.squeeze(gamma_0),np.squeeze(alpha_minus),np.squeeze(beta_minus)], dtype=np.float64 )
  positions = np.array([-M_bht, -1, 0, 1, M_bht], dtype=np.float64 )
  
  # create matrix A
  A = spdiags( rows, positions, JM_bht, JM_bht )
  A = A.T
  A = A*K - P
  
  # create matrix B
  B = eye(JM_bht,JM_bht) - 0.5*dt*A
  
  del alpha_plus,alpha_minus,bp,bm,beta_plus,beta_minus,gamma_0,k1minor,k2minor,k,bigk,p1,p2,pp,bigp,P,rows,positions
   
  # accelerate algorithm by prefactoring
  solve1 = factorized(A)
  solve2 = factorized(B)

  # rescale H_bht to degrees/second
  H_bht[:,0:m_-1]     = 1e6*H_bht[:,0:m_-1]/Cv1/rho1
  H_bht[:,m_:M_bht-1] = 1e6*H_bht[:,m_:M_bht-1]/p[1]

  # temperature vector
  T = np.zeros( (JM_bht,1), dtype=np.float64 )
  
  # column-stack the heat matrix Q, converts JxK matrix into a column-stacked J*Kx1 vector
  Q = np.squeeze( H_bht.reshape(-1,JM_bht).transpose() )

  # slopes for implicit Runge-Kutta integrator
  s1 = np.zeros( (JM_bht,1), dtype=np.float64 )
  s2 = np.zeros( (JM_bht,1), dtype=np.float64 )

  # peak temperature as a function of time
  Tpeak = np.zeros( (N,1), dtype=np.float64 )

  # dummy peak temperature
  tt = -0.1

  # computation of solution  
  #-----------------------------
  
  # time stepping
  for n in np.arange(0,N-1):
    # set as integer
    n = int(n)
    # squeeze as vector
    sequence = np.squeeze(sequence)
    T        = np.squeeze(T)
    # compute slopes of Runge-Kutta scheme
    s1 = A.dot(T) 
    s1 += np.squeeze(sequence[n]*Q)
    if (ilu==1):
      s2 = solve2( A.dot(T+0.5*dt*s1) + np.squeeze(sequence[n+1]*Q) )
    else:
      s2 = spsolve( B, A.dot(T+0.5*dt*s1) + np.squeeze(sequence[n+1]*Q) )
    # update temperature vector
    T = T + 0.5*dt*(s1+s2)
    # get spatial peak temperature
    Tpeak[n] = np.max(T)
  
  err = ydata - np.squeeze(Tpeak)  
  
  iterate = iterate + 1 
  str1 = '{:.7e}'.format( p[0]*p[1]/(1.0e4) )
  str2 = '{:.7e}'.format( p[1]*1070.0/1e6 )
  str3 = '{:.7e}'.format( np.sum( np.abs( err ) ) )
  print '\t' + str1 + '\t' + str2 + '\t' + str3
            
  return err   

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

# clear screen
os.system('cls' if os.name == 'nt' else 'clear')

# start timer
printstart = colored(' starting ...', 'blue')
ts = time.time()
print datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S' + printstart)

tstart = time.time()

# dictionary containing dBm levels and powers.
dbm = { \
'-22':  1.84, \
'-20':  2.91, \
'-19':  3.67, \
'-18':  4.61, \
'-16':  7.31, \
'-15':  9.21, \
'-14': 11.59, \
'-13': 14.60, \
'-12': 18.38, \
'-11': 23.14, \
'-10': 29.13, \
'-9' : 36.67, \
'-8' : 46.17, \
'-7' : 58.13, \
'-6' : 73.18, \
'-5' : 92.13}

drive_level = 16

drive_setting = '-' + str(drive_level)

# drive power [Watts]
drive = dbm[drive_setting] 

# efficiency of transducer [dimensionless]
efficiency = 0.75

# acoustic data with heating rate, H, and axisymmetric acoustic domain sizes z,r
acoustic_filename='cryogel_' + drive_setting + '.mat'

# generate acoustic field
z,r,H,I,Ppos,Pneg,Ix,Ir,p0,p5r,p5x,peak,trough,rho1,rho2,c1,c2,R,d,Z,M,a,m_t,f,Xpeak,z_peak,K2,z_ = axisymmetricKZK.axisymmetricKZK(drive,efficiency)

# rescaled peak-positive and peak negative pressures
peak   = 1e-6*p0*peak
trough = 1e-6*p0*trough

# safety check for length of vectors for plotting  
Nlength = np.min( np.array([np.shape(np.squeeze(z))[0], np.shape(np.squeeze(trough))[0], np.shape(np.squeeze(peak))[0]]) )

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
fig0, ax0 = plt.subplots()
fig0.hold(True)
ax0.set_xlim([0,Z*d])
xticks0 = np.arange(0,Z*d,10)
ax0.set_xlabel(r'z [cm]', fontsize=14, color='black')
ax0.set_ylabel(r'Pressure [MPa]', fontsize=14, color='black')
ax0.set_title(r'Axial Pressures', fontsize=14, color='black')
ax0.set_xticks(xticks0,minor=True)  
ax0.minorticks_on()
ax0.plot( np.squeeze(z)[0:Nlength], np.squeeze(np.transpose(peak))[0:Nlength], linewidth=2, linestyle='-', color=ICRcolors['ICRred'] )
ax0.plot( np.squeeze(z)[0:Nlength], np.squeeze(np.transpose(trough))[0:Nlength], linewidth=2, linestyle='-', color=ICRcolors['ICRgreen'] )
plt.grid(True)

del I,Ppos,Pneg,Ix,Ir,p0,p5r,p5x,peak,trough,rho1,rho2,c1,c2,R,d,Z,M,a,m_t,f,Xpeak,z_peak,K2,z_ 

# save data in matlab format
savemat(acoustic_filename, {'H':H, 'z':z, 'r':r} )

# heat capacity [J/kg/K]
Cv2 = 3750.0

# thermal conductivity [W/m/K]
k2 = 0.58   

# perfusion rate [kg/m^3/s] : Med. Phys. (2000) 27(5) p.1131-1140 use 0.5-20
w2 = 0.0   

# mass density [kg/m^3]
rho2 = 1070.0 

# input parameters
pstart = [ 1e4*k2/(Cv2*rho2), Cv2*rho2  ]

# tolerances
tol = np.array( [1e-06, 1e-06] )

# experimental data
experimental_filename='20dBm_1_0_0_0_20140516_4s_1c.txt'

# iterate
iterate = 0

# formatting of output to print to screen
print '\t' + 'kappa' + '\t\t' + 'C_v' + '\t\t' + 'Residue'
print '\t' + '-------------' + '\t' + '-------------' + '\t' + '-------------'

# least squares minimization
pfin, cov_x, infodict, mesg, ierr = leastsq(func, pstart, args=(acoustic_filename, experimental_filename, iterate, drive, efficiency), Dfun=None, full_output=1, col_deriv=0, ftol=tol[0], xtol=tol[1] )

elapsed = time.time() - tstart
hours = np.floor(elapsed/3600.0)
minutes = np.floor( (elapsed - hours*3600.0)/60.0 )
seconds = elapsed - 60.0*(hours*60.0 + minutes)
printfin = colored('\n... finished', 'blue')
if (hours >= 1):
  if (hours >1):
    if (minutes > 1):
      print printfin + ' in {0} hours, {1} minutes and {2} seconds\n'.format(int(hours), int(minutes), seconds)
    if (minutes == 1):
      print printfin + ' in {0} hours, {1} minute and {2} seconds\n'.format(int(hours), int(minutes), seconds)
    if (minutes < 1):
      print printfin + ' in {0} hours and {1} seconds'.format(int(hours), seconds)
  else:
    if (minutes > 1):
      print printfin + ' in {0} hour, {1} minutes and {2:.2%} seconds\n'.format(int(hours), int(minutes), seconds)   
    if (minutes == 1):
      print printfin + ' in {0} hour, {1} minute and {2:.2%} seconds\n'.format(int(hours), int(minutes), seconds)
    if (minutes < 1):
      print printfin + ' in {0} hour and {1} seconds\n'.format(int(hours), seconds)
if ((hours < 1) and (minutes >= 1)):
  if (minutes == 1):        
    print printfin + ' in {0} minute and {1} seconds\n'.format(int(minutes), seconds)
  else:
    print printfin + ' in {0} minutes and {1:0.2f} seconds\n'.format(int(minutes), seconds)
if ((hours < 1) and (minutes < 1)):    
  print printfin + " in %5.3f seconds.\n" %(seconds)

# if success print parameters and get graph.
if ( (ierr == 1) or (ierr == 2) or (ierr == 3)  or (ierr == 4) ):  
  print "ierr = ", ierr, ": Success! \n"
  print "kappa2, Cv = {0:0.7f}, {1:0.7f}".format(pfin[0]*pfin[1]/(1.0e4), pfin[1]*1070.0/1e6)
  print "lit values = {0:0.7f}, {1:0.7f}\n".format(pstart[0]*pstart[1]/(1.0e4), pstart[1]*1070.0/1e6)
  # check whether file exists
  if os.path.isfile(acoustic_filename):
    # load data
    matlab_format_data = loadmat(acoustic_filename, variable_names=['z','r','H'] )
    z = matlab_format_data['z']
    r = matlab_format_data['r']
    H = matlab_format_data['H']
  else:
    print "No file to load in this instance!"
    exit(1)
    
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
  
  # automated start-time detection
  jtstart = np.argmax( np.gradient(data[:,1]) )

  # automated detection of when transducer is switched off
  jtoff = np.argmax( data[:,1] ) 

  # specify duration of relevant recorded cool-off period, in this case 10 seconds
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
  
  # interpolate to get approximate temperatures at channel switching points 
  jump = np.interp( channelswitch[:,0], data[:,0], data[:,1],  )
      
  # index of last relevant time point
  jtend = np.argmin( np.abs(channelswitch[:,0] - Tend) )
  
  # set start time to zero seconds
  channelswitch = channelswitch[0:jtend,0]- t[itstart]
  jump = jump[0:jtend] - y[itstart]
  
  # determine unique instances at which laser is retuning
  diff = np.zeros( ( np.shape(jump)[0]-1,) )
  time_channel_switch = []
  temp_channel_switch = []
  for i in np.arange(0, np.int( np.shape(jump)[0] )-1 ):
    diff[i] = jump[i] - jump[i+1]
    if (np.abs(diff[i]) > 0.02):
      time_channel_switch = np.append(time_channel_switch, channelswitch[i])
      temp_channel_switch = np.append(temp_channel_switch, jump[i])
  
  # get index which specifies last relevant time point
  itend = np.argmin( np.abs(t - Tend) )

  # sampled experimental data
  tdata = t[itstart:itend] - t[itstart]
  ydata = y[itstart:itend] - y[itstart]

  # start time [seconds]
  tstart = tdata[0]

  # end time [seconds]
  tend = tdata[-1]

  # index at which transducer is switched off
  itoff = np.argmax( ydata ) 

  # time at which transducer is switched off [seconds]
  toff = tdata[ itoff ]

  # get acoustic parameters
  p0,c1,c2,rho1,rho2,N1,N2,G1,G2,gamma1,gamma2,alpha1,alpha2,a,b,d,f,R,Z,z_,K = KZK_parameters.KZK_parameters(drive,efficiency)

  # get thermal parameters:
  C1,C2,k1,k2,w1,w2,N,t,sequence,dt,J_bht,M_bht,z_bht,r_bht,H_bht,T0 = BHT_parameters.BHT_parameters(z,r,H,tstart,tend,toff)

  #------------------
  del sequence, dt, N
  #------------------ 
  
  # output progress to screen
  print '\nIntegrating BHT equation...\n'
  print '\tt [sec]\t\ttime [hr:min:sec]\tn'

  # determine proper timestep:
  dt = tdata[1]

  # number of time points
  N = np.shape(tdata)[0]

  # duration of exposure
  sequence = np.zeros( (N+1,1) )
  for i in np.arange(0, itoff+1):
    sequence[i] = 1.0  
    
  # nondimensional diffusivity of material 1 [dimensionless]
  D1 = 1e4*k1/C1/rho1

  # nondimensional diffusivity of material 2 [dimensionless]
  D2 = pfin[0]

  # nondimensional perfusion/density of material 1 [dimensionless]
  P1 = w1/rho1

  # nondimensional perfusion/density of material 2 [dimensionless]
  P2 = 0.0

  # create Crank-Nicolson operators for BHT
  A,B = BHT_operators.BHT_operators(z_bht,r_bht,D1,D2,P1,P2,dt,z_/d/Z)
  solve1 = factorized(A)
  solve2 = factorized(B)

  # index at material interface 
  m_ = np.around(z_*M_bht/d/Z)
  if (m_>M_bht): 
    m_ = M_bht 
    
  # rescale H_bht to degrees/second
  H_bht[:,0:m_-1] = 1e6*H_bht[:,0:m_-1]/C1/rho1
  H_bht[:,m_:M_bht-1] = 1e6*H_bht[:,m_:M_bht-1]/pfin[1]

  # define vector size
  JM_bht = J_bht*M_bht

  # temperature vector
  T = np.zeros( (JM_bht,1), dtype=np.float64 )

  # thermal dose vector
  D = np.zeros( (JM_bht,1), dtype=np.float64 )

  # column-stack the heat matrix Q, converts JxK matrix into a column-stacked J*Kx1 vector
  Q = np.squeeze( H_bht.reshape(-1,JM_bht).transpose() )

  # slopes for implicit Runge-Kutta integrator
  s1 = np.zeros( (JM_bht,1), dtype=np.float64 )
  s2 = np.zeros( (JM_bht,1), dtype=np.float64 )

  # peak temperature as a function of time
  Tpeak = np.zeros( (N+1,1), dtype=np.float64 )

  # dummy peak temperature
  tt = -0.1
    
  steps = np.floor( np.linspace(0,N-2,10) )
  steps = steps.astype(int)

  # initialise timer
  t_start = time.time()

  # dummy time variable
  p1 = 0

  # time stepping
  for n in np.arange(0,N):
    # set as integer
    n = int(n)
    # squeeze as vector
    sequence = np.squeeze(sequence)
    T = np.squeeze(T)
    # compute slopes of Runge-Kutta scheme
    s1 = A.dot(T) 
    s1 += np.squeeze(sequence[n]*Q)
    s2 = solve2( A.dot(T+0.5*dt*s1) + np.squeeze(sequence[n+1]*Q) )  
    # update temperature vector
    T = T + 0.5*dt*(s1+s2)
    # get spatial peak temperature
    Tpeak[n+1] = np.max(T)
    # check whether spatial peak temp is maximal
    if (Tpeak[n+1]>tt):
      tt = np.float(Tpeak[n+1])
      Tmax_vec = T
      t_peak = t[n+1]
    # output to screen
    if n in steps:
      p2 = np.floor( 10.0*(n+1.0)/N )
      p1 = timing.timing( p1, p2, t_start, t[n], 1.0, n )
  print "\n"
  
  #-------------------------------
    
  # nondimensional diffusivity of material 2 [dimensionless]
  D2 = pstart[0]

  # create Crank-Nicolson operators for BHT
  A,B = BHT_operators.BHT_operators(z_bht,r_bht,D1,D2,P1,P2,dt,z_/d/Z)
  solve1 = factorized(A)
  solve2 = factorized(B)
    
  # rescale H_bht to degrees/second
  H_bht[:,m_:M_bht-1] = pfin[1]*H_bht[:,m_:M_bht-1]/pstart[1]

  # temperature vector
  T = np.zeros( (JM_bht,1), dtype=np.float64 )

  # column-stack the heat matrix Q, converts JxK matrix into a column-stacked J*Kx1 vector
  Q = np.squeeze( H_bht.reshape(-1,JM_bht).transpose() )

  # slopes for implicit Runge-Kutta integrator
  s1 = np.zeros( (JM_bht,1), dtype=np.float64 )
  s2 = np.zeros( (JM_bht,1), dtype=np.float64 )

  # peak temperature as a function of time
  Tpeak0 = np.zeros( (N+1,1), dtype=np.float64 )

  # time stepping
  for n in np.arange(0,N):
    # set as integer
    n = int(n)
    T = np.squeeze(T)
    # compute slopes of Runge-Kutta scheme
    s1 = A.dot(T) 
    s1 += np.squeeze(sequence[n]*Q)
    s2 = solve2( A.dot(T+0.5*dt*s1) + np.squeeze(sequence[n+1]*Q) )  
    # update temperature vector
    T = T + 0.5*dt*(s1+s2)
    # get spatial peak temperature
    Tpeak0[n+1] = np.max(T)  
    
  #-------------------------------
    
  # define figure and axis handles
  fig1, ax1 = plt.subplots()
  # hold on for multiple plots on figure
  fig1.hold(True)
  # apply grid to figure
  plt.grid(True)
  # plot experimental temperatures
  plot1, = ax1.plot( tdata, ydata, linewidth=2, linestyle='-', color=ICRcolors['ICRred'], label='Experimental' )
  ax1.plot( tdata, ydata, marker='o', color=ICRcolors['ICRred'] )
  # plot optimized temperatures
  plot2, = ax1.plot( tdata, np.squeeze(Tpeak[0:N]), linewidth=2, linestyle='-', color=ICRcolors['ICRblue'], label='Least Squares' )
  ax1.plot( tdata, np.squeeze(Tpeak[0:N]), marker='o', color=ICRcolors['ICRblue'] )
  # plot book value temperatures
  plot3, = ax1.plot( tdata, np.squeeze(Tpeak0[0:N]), linewidth=2, linestyle='-', color=ICRcolors['ICRgreen'], label='Book Values' )
  ax1.plot( tdata, np.squeeze(Tpeak0[0:N]), marker='o', color=ICRcolors['ICRgreen'] )
  for i in np.arange(0,np.shape(temp_channel_switch)[0]-1):
    ax1.plot( time_channel_switch[i], temp_channel_switch[i], marker='+', color=ICRcolors['ICRblue'] )
    plt.axvline(time_channel_switch[i], color='black')
    plt.axhline(temp_channel_switch[i], color='black')
  # define xticks
  xticks1 = np.linspace( np.squeeze(tdata[0]), np.squeeze(tdata[-1]), num=10)
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
  # set legend for plot
  # set minor ticks on both axes
  ax1.minorticks_on()
  # set legend for plot
  ax1.legend( [plot1, plot2, plot3], [r'Experimental', r'Least Squares', r'Book Values'] )

  # define figure and axis handles
  fig2, ax2 = plt.subplots()
  # hold on for multiple plots on figure
  fig2.hold(True)
  # plot peak temperature
  plot4, = ax2.plot( tdata, ydata-np.squeeze(Tpeak[0:N]), linewidth=2, linestyle='-', color=ICRcolors['ICRred'])
  ax2.plot( tdata, ydata-np.squeeze(Tpeak[0:N]), marker='o', color=ICRcolors['ICRred'] )
  # plot points of channel switching
  for i in np.arange(0,np.shape(temp_channel_switch)[0]-1):
    plt.axvline(time_channel_switch[i], color='black')
  #plot4, = ax2.plot( tdata, np.abs(ydata-np.squeeze(Tpeak[0:N])), linewidth=2, linestyle='-', color=ICRcolors['ICRgreen'])
  #ax2.plot( tdata, np.abs( ydata-np.squeeze(Tpeak[0:N]) ) , marker='o', color=ICRcolors['ICRgreen'] )
  # define xticks
  xticks2 = np.linspace( np.squeeze(tdata[0]), np.squeeze(tdata[-1]), num=10)
  # define xlabel
  ax2.set_xlabel(r'$t$ [sec]', fontsize=14, color='black')
  # define ylabel
  ax2.set_ylabel(r'$\epsilon$ [Degrees]', fontsize=14, color='black')
  # set title
  ax2.set_title(r'Difference', fontsize=14, color='black')
  # set limits of x-axis
  ax2.set_xlim( [np.squeeze(tdata[0]), np.squeeze(tdata[-1])] )
  # apply xticks
  ax2.set_xticks(xticks2,minor=True)
  # set minor ticks on both axes
  ax2.minorticks_on()
  # apply grid to figure
  plt.grid(True)
  # plot x zero axis
  plt.axhline(0, color='black')
  
  # render figures
  plt.show()   
  
  # get stem of output filename from experimental_filename
  outfile0, fileExtension = os.path.splitext(experimental_filename)
  # add prefix and suffix to filename
  outfile = 'data_'+outfile0+'.out'
  # add header which contains converged parameters
  headertxt = str(pfin[0]) + ', ' + str(pfin[1])
  # check whether filename exists
  if os.path.isfile(outfile):
    warning = colored(" *** file already exists *** ", 'red')
    print( warning )
    answer = raw_input('Enter Y to write to file : ')
    if ( (answer == 'Y') or (answer == 'y') ):
      # write to file
      with open(outfile,'a') as f_handle:
	np.savetxt( f_handle, [pfin[0], pfin[1]], fmt='%1.7e', newline='\n' )
	np.savetxt( f_handle, np.transpose( np.vstack( (tdata, np.squeeze(Tpeak[0:N])) ) ), fmt='%1.4e', newline='\n' ) 
    else:
      print " File not saved. Exiting."
      print " "
  else:
    # write to file
    with open(outfile,'w') as f_handle: 
      np.savetxt( f_handle, [pfin[0], pfin[1]], fmt='%1.7e', newline='\n' )
      np.savetxt( f_handle, np.transpose(np.vstack( (tdata, np.squeeze(Tpeak[0:N])) ) ), fmt='%1.4e', newline='\n' )
  
else:
  print "ierr = ", ierr, ": Failure \n"
  print mesg