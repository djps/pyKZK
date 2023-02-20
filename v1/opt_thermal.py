"""

leastsq is a wrapper around MINPACK's lmdif and lmder algorithms.

covx is a Jacobian approximation to the Hessian of the least squares objective function. This 
approximation assumes that the objective function is based on the difference between some 
observed target data (ydata) and a (non-linear) function of the parameters f(xdata, params)

"""

#--------------------------------------------------------------------------------------------------
    
def func(p, acoustic_filename, experimental_filename):

  #-------------------------------------------------
  # preamble, warnings and information, clear screen
  #-------------------------------------------------
  
  import numpy as np
  import os, time, datetime, warnings

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
  
  # plot data
  if (ifig==1):
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
    # plot peak temperature
    ax1.plot( tdata, ydata, linewidth=2, linestyle='-', color=ICRcolors['ICRred'] )
    #ax1.plot( t[itstart], y[itstart], marker='*', color=ICRcolors['ICRblue'] )
    ax1.plot( data[jtstart:jtend,0], data[jtstart:jtend,1], linewidth=1, linestyle='-.', color=ICRcolors['ICRgreen'] )
    ax1.plot( data[jtstart,0], data[jtstart,1], marker='o', color=ICRcolors['ICRolive'] )
    # define xticks
    #xticks = np.arange( t[itstart],t[itend],10)
    # define xlabel
    ax1.set_xlabel(r'$t$ [sec]', fontsize=14, color='black')
    # define ylabel
    ax1.set_ylabel(r'$T$ [Degrees]', fontsize=14, color='black')
    # set title
    ax1.set_title(r'Peak Temperature', fontsize=14, color='black')
    # apply xticks
    #ax1.set_xticks(xticks,minor=True)
    # set minor ticks on both axes
    ax1.minorticks_on()
    # apply grid to figure
    plt.grid(True)
    # render figure
    plt.show()
  
  # Drive settings
  #-----------------------------
  
  # drive power [Watts]
  drive = 3.0

  # efficiency of transducer [dimensionless]
  efficiency = 0.75

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
  for i in np.arange(0, itoff):
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
  P2 = p[1]

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
  H_bht[:,m_:M_bht-1] = 1e6*H_bht[:,m_:M_bht-1]/p[2]

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
            
  return err   

#--------------------------------------------------------------------------------------------------
# preamble, warnings and information, clear screen
  
from scipy.optimize import leastsq  
import os, time, datetime, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from math import pi, log

from scipy.sparse.linalg import spsolve, factorized, splu
from scipy.linalg import solve, lu

from scipy.io import loadmat, savemat

import equivalent_time, timing, KZK_parameters, BHT_parameters, BHT_operators

from scipy.sparse import SparseEfficiencyWarning

warnings.simplefilter("ignore", SparseEfficiencyWarning)

# clear screen
os.system('cls' if os.name == 'nt' else 'clear')

# heat capacity (J/kg/K)
Cv2 = 3750.0

# thermal conductivity [W/m/K]
k2 = 0.58   

# perfusion rate [kg/m^3/s] : Med. Phys. (2000) 27(5) p.1131-1140 use 0.5-20
w2 = 0.0   

# mass density [kg/m^3]
rho2    = 1070.0 

# input parameters
p0 = [ 1e4*k2/(Cv2*rho2), w2/rho2, Cv2*rho2  ]

tol = np.array( [1e-06, 1e-06] )

experimental_filename='20dBm_1_0_0_0_20140516_4s_1c.txt'

acoustic_filename='cryogel_20.mat'

# least squares minimization
pfin, cov_x, infodict, mesg, ierr = leastsq(func, p0, args=(acoustic_filename, experimental_filename), Dfun=None, full_output=1, col_deriv=0, ftol=tol[0], xtol=tol[1] )

# if success print parameters and get graph.
if ( (ierr != 1) or (ierr != 2) or (ierr != 3)  or (ierr != 4) ):  
  print "ierr = ", ierr, " Success! \n"
  print "kappa2, w2, Cv = ", pfin[0]/(1.0e4*pfin[2]), pfin[1]*1070.0, pfin[2]*1070.0, "\n"
  print mesg
    
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
      
  # drive power [Watts]
  drive = 3.0

  # efficiency of transducer [dimensionless]
  efficiency = 0.75

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
  for i in np.arange(0, itoff):
    sequence[i] = 1.0  
    
  # nondimensional diffusivity of material 1 [dimensionless]
  D1 = 1e4*k1/C1/rho1

  # nondimensional diffusivity of material 2 [dimensionless]
  D2 = pfin[0]

  # nondimensional perfusion/density of material 1 [dimensionless]
  P1 = w1/rho1

  # nondimensional perfusion/density of material 2 [dimensionless]
  P2 = pfin[1]

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
  H_bht[:,m_:M_bht-1] = 1e6*H_bht[:,m_:M_bht-1]/pfin[2]

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
  
  # plot peak temperature
  ax1.plot( tdata, ydata, linewidth=2, linestyle='-', color=ICRcolors['ICRred'] )
  ax1.plot( tdata, np.squeeze(Tpeak[0:N]), linewidth=2, linestyle='-', color=ICRcolors['ICRblue'] )
  
  # define xticks
  xticks = np.linspace( np.squeeze(tdata[0]), np.squeeze(tdata[-1]), num=10)
  
  # define xlabel
  ax1.set_xlabel(r'$t$ [sec]', fontsize=14, color='black')
  
  # define ylabel
  ax1.set_ylabel(r'$\Delta T$ [Degrees]', fontsize=14, color='black')
  
  # set title
  ax1.set_title(r'Peak Temperature Rise', fontsize=14, color='black')
  
  # apply xticks
  ax1.set_xticks(xticks,minor=True)
  
  # set minor ticks on both axes
  ax1.minorticks_on()
  
  # apply grid to figure
  plt.grid(True)
  
  # render figure
  plt.show()   