import numpy as np
import os, time, datetime

from math import pi, log

from scipy.sparse.linalg import spsolve
from scipy.linalg import solve, lu

#import peakdetect, intersections

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from scipy.io import loadmat, savemat

import equivalent_time, timing, KZK_parameters, BHT_parameters, BHT_operators, axisymmetricKZK

#--------------------------------------------------------------------------------------------------
# preamble, warnings and information, clear screen

# clear screen
os.system('cls' if os.name == 'nt' else 'clear')

#--------------------------------------------------------------------------------------------------

iverbose=0

# drive power [Watts]
drive = 19.0

# efficiency of transducer [dimensionless]
efficiency = 0.75

# start time [seconds]
tstart = 0.0

# end time [seconds]
tend = 15.0

# time at which transducer is switched off [seconds]
toff = 5.0

# z,r,H,I,Ppos,Pneg,Ix,Ir,p0,p5r,p5x,peak,trough,rho1,rho2,c1,c2,R,d,Z,M,a,m_t,f,Xpeak,z_peak,K2,z_ = axisymmetricKZK.axisymmetricKZK(drive,efficiency)

#filename = "../../../../Code/least squares bioheat/Madden/Axi/Computed_Data/thermal_5s_-11dbm_1.mat"
filename=os.path.expanduser('~/Desktop/Thermal_pycomp_5s_-12dbm_1.mat')

# load canonical data set, but only load what is strictly necessary if iload=0, else load all.
iload = 0
if iload==1:
  data = loadmat(filename)
  z = data['z']
  r = data['r']
  H = data['H']
  A = data['A']
  B = data['B']
  A.sort_indices()
  B.sort_indices()
  if iverbose==1:
    print dir(data), type(data)
    for (key,value) in data.iteritems() :
      print key
else:
  data = loadmat(filename, variable_names=['z','r','H'] )
  z = data['z']
  r = data['r']
  H = data['H']
  
# get acoustic parameters
p0,c1,c2,rho1,rho2,N1,N2,G1,G2,gamma1,gamma2,alpha1,alpha2,a,b,d,f,R,Z,z_,K = KZK_parameters.KZK_parameters(drive,efficiency)

# get thermal parameters:
C1,C2,k1,k2,w1,w2,N,t,sequence,dt,J_bht,M_bht,z_bht,r_bht,H_bht,T0 = BHT_parameters.BHT_parameters(z,r,H,tstart,tend,toff)

# nondimensional diffusivity of material 1 [dimensionless]
D1 = 1e4*k1/C1/rho1

# nondimensional diffusivity of material 2 [dimensionless]
D2 = 1e4*k2/C2/rho2

# nondimensional perfusion/density of material 1 [dimensionless]
P1 = w1/rho1

# nondimensional perfusion/density of material 2 [dimensionless]
P2 = w2/rho2

# create Crank-Nicolson operators for BHT
manual=1
ilu=0
if manual==1:
  A,B = BHT_operators.BHT_operators(z_bht,r_bht,D1,D2,P1,P2,dt,z_/d/Z)
  if ilu==1:
    PD, L, U = lu(A2.todense())
    PD, L, U = lu(B2.todense())

#if ((iverbose==1) and (manual==1)):
  #print type(A), type(A2), dir(A), dir(A2)
  #print np.shape(B), np.shape(B2)
  #Atemp = A.toarray()
  #A1temp = A1.todense()
  #print "\tA", A[0:15,0:15], '\n'
  #print "\tA1", A2[0:15,0:15], '\n'
  #print "\tB", B[0:15,0:15], '\n'
  #print "\tB1", B2[0:15,0:15], '\n'
  #print np.shape(B.data),np.shape(B2.data)
  #print np.shape(B.indices), np.shape(B2.indices)
  #print np.shape(B.indptr), np.shape(B2.indptr)
  #print np.shape(A.data),np.shape(A2.data)
  #print np.shape(A.indices), np.shape(A2.indices)
  #print np.shape(A.indptr), np.shape(A2.indptr)

# index at material interface 
m_ = np.around(z_*M_bht/d/Z)
if (m_>M_bht): 
   m_ = M_bht 
   
if iverbose==1:
  print m_, dt
   
# rescale H_bht to degrees/second
H_bht[:,0:m_-1] = 1e6*H_bht[:,0:m_-1]/C1/rho1
H_bht[:,m_:M_bht-1] = 1e6*H_bht[:,m_:M_bht-1]/C2/rho2

# vector size
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

# temp vs time vector
Tpeak = np.zeros( (N+1,1), dtype=np.float64 )

# dummy peak temperature
tt = -0.1

# max dose
Dmax = np.zeros( (N+1,1), dtype=np.float64 )

if iverbose==1:
  print sequence

#--------------------------------------------------------------------------------------------------

# Integrate BHT:
print '\nIntegrating BHT equation...\n'
print '\tt [sec]\t\ttime [hr:min:sec]\tn'

# initialise timer
t_start = time.time()

# dummy time variable
p1 = 0

# input time index
p2 = np.floor( 10.0/N )

# timer output
p1 = timing.timing( p1, p2, t_start, 0.0, 1.0, np.int(0) )

# time stepping
for n in np.arange(0,N-1):

  # set as integer
  n = int(n)
  
  # squeeze as vector
  sequence = np.squeeze(sequence)
  
  #if (n != 0):
    #T = np.transpose(np.expand_dims(T, axis=0))

  T = np.squeeze(T)
   
  #print type( A*T + sequence[n]*Q ), type(B), type(A*(T+s1) + sequence[n+1]*Q)
  #print type(A), type(B), type(T), type(Q), type(sequence[n]), type(sequence[n]*Q)
  #print np.shape(A), np.shape(B), np.shape(T), np.shape(Q), np.shape(sequence[n]), np.shape(sequence[n]*Q)
  #intermed = A.dot(T)
  #print type(intermed)
  s1 = A.dot(T) 
  #print type(s1), np.shape(s1)
  s1 += np.squeeze(sequence[n]*Q)
  #print type(s1)
  
  #print np.shape(T), np.shape(s1), np.shape( np.squeeze(sequence[n+1]*Q) )
  s2 = spsolve( B, A.dot(T+0.5*dt*s1) + np.squeeze(sequence[n+1]*Q) )
  
  #T = np.squeeze(T)
  T = T + 0.5*dt*(s1+s2)
  #T += np.squeeze(0.5*dt*s2)
  
  Tpeak[n+1] = np.max(T)
  
  D = np.squeeze(D)
  D += equivalent_time.equivalent_time( T, JM_bht )
  
  Dmax[n+1]  = np.amax(D)

  if (Tpeak[n+1]>tt):
    tt = np.float(Tpeak[n+1])
    Tmax_vec = T
    t_peak = t[n+1]
  
  if ( (np.mod(n,10) == 0) ): #or (n==int(0)) or (n==int(N-1)) ):
    p2 = np.floor( 10.0*(n+1.0)/N )
    p1 = timing.timing( p1, p2, t_start, t[n], 1.0, n )
    
  #if iverbose==1:
  #print t[n], sequence[n], tt, np.amax(s1), np.amax(s2), np.max(Q), '\n'
    
  # compute minimum time at which a thermal dose of delivered.
  if (np.min( np.abs( np.squeeze(Dmax[:,0]) - 240.0 ) ) < 0.001 ):
    min_itime = np.argmin( np.abs( Dmax[:,0] - 240.0 ) )
    min_time = t[min_itime]
    print '\n\tBeginning to lesion ... \t %f \n', min_time
    
#--------------------------------------------------------------------------------------------------

p2 = np.floor( 10.0*(n+1.0)/N )
p1 = timing.timing( p1, p2, t_start, t[n], 1.0, n )

print "\n"

#--------------------------------------------------------------------------------------------------
    
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
  
fig1, ax = plt.subplots()
fig1.hold(True)
ax.plot( np.squeeze(t[0:N]), np.squeeze(np.transpose(Tpeak[0:N])), linewidth=2, linestyle='-', color=ICRcolors['ICRred'] )
xticks = np.arange(0,tend,10)
ax.set_xlabel('t [sec]', fontsize=14, color='black')
ax.set_ylabel('T [Degrees]', fontsize=14, color='black')
ax.set_title('Peak Temperature', fontsize=14, color='black')
ax.set_xticks(xticks,minor=True)
plt.grid(True)
plt.show()
