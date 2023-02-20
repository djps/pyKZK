'''
  runner.py
'''

import numpy as np

import os, time, datetime

import axisymmetricKZK, axisymmetricBHT

from scipy.io import savemat, loadmat

# clear screen
os.system('cls' if os.name == 'nt' else 'clear')

#exper_perfus = ...
     #{'Away_from_vessel/Exp2/5s_-11dbm_1.txt', 23.14, 0.74, 1.39, 1.54, 0.12, 0.10, 0.99, 0.11; ...
      #'Away_from_vessel/Exp2/5s_-12dbm_1.txt', 18.38, 0.74, 1.89, 2.27, 0.15, 0.20, 0.99, 0.15; ...
      #'Away_from_vessel/Exp2/5s_-13dbm_1.txt', 14.60, 0.74, 2.58, 3.34, 0.20, 0.50, 0.98, 0.20; ...
      #'Away_from_vessel/Exp2/5s_-16dbm_1.txt', 7.31, 0.74, 3.52, 4.92, 0.28,	0.90, 0.96, 0.27; ...
      #'Away_from_vessel/Exp2/5s_-19dbm_1.txt', 3.67, 0.74, 3.91, 5.59, 0.30,	0.11, 0.95, 0.29; ...
      #'Away_from_vessel/Exp2/5s_-22dbm_1.txt', 1.84, 0.74, 4.33, 6.36, 0.34,	0.13, 0.94, 0.31};

## specify which data set
#entry = 1

## read filename, drive level and effciency
#filename = exper_perfus{entry,1}
#drive = exper_perfus{entry,2}
#efficiency = exper_perfus{entry,3}

## separate name and extension
#[dummy, fname, ext] = fileparts(filename);

drive = 3.0
efficiency = 0.74

# compute acoustic field
z,r,H,I,Ppos,Pneg,Ix,Ir,p0,p5r,p5x,peak,trough,rho1,rho2,c1,c2,R,d,Z,M,a,m_t,f,Xpeak,z_peak,K2,z_ = axisymmetricKZK.axisymmetricKZK(drive,efficiency)

filename='cryogel_20.mat'
#savemat(filename, {'H':H, 'z':z, 'r':r} )

#print "\n"

# set location of filename

## check whether file exists
#if os.path.isfile(filename):
  ## load canonical data set, but only load what is strictly necessary if iload=0, else load all.
  #if iload==1:
data = loadmat(filename)
z = data['z']
r = data['r']
H = data['H']
    #A = data['A']
    #B = data['B']
    #A.sort_indices()
    #B.sort_indices()
    #if iverbose==1:
      #print dir(data), type(data)
      #for (key,value) in data.iteritems() :
	#print key
  #else:
    ## load minimum ammount
    #data = loadmat(filename, variable_names=['z','r','H'] )
    #z = data['z']
    #r = data['r']
    #H = data['H']
#else:
  #print "No file to load in this instance!"

axisymmetricBHT.axisymmetricBHT(H, z, r)

## visualise volumes
##converter(Z, R, d, a, z, r, z_bht,r_bht, p5x, p5r,p0, Pneg, Ppos, peak,trough, Dmat, Tmax_mat);
