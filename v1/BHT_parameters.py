def BHT_parameters(z,r,H,tstart,tend,toff):
  
  """
  
  this function determines the values of the parameters used in BHT
  equation. The main point is that every thing which was J,M -> J_bht etc.
  
  """
  
  import numpy as np
  from math import pi, log

  #---------------------------- User-defined input parameters -----------------------------
 
  # material 1: WATER
  #--------------------#
  
  # heat capacity (J/kg/K)
  C1 = 4180.0 
  
  # thermal conductivity (W/m/K)
  k1 = 0.6  
  
  # perfusion rate (kg/m^3/s)
  w1 = 0.0  

  # material 2: LIVER
  #--------------------#
  
  # heat capacity (J/kg/K)
  C2 = 3750.0
  
  # thermal conductivity (W/m/K)
  k2 = 0.58   
  
  # perfusion rate (kg/m^3/s) : [Med. Phys. (2000) 27(5) p.1131-1140] use 0.5-20
  w2 = 20.0   

  # ambient temperature (degrees C)
  T0 = 37.0

  # initial sonication duration (s)
  t_i = toff

  # periodic sonications
  #--------------------#
  
  # number of pulse cycles
  n_c = 0
  
  # duty cycle (#)
  D   = 0.0
  
  # pulse cycle period (s)
  t_p = 0.0
  
  # cooling period
  t_c = tend - tstart - toff

  # grid coarsening
  #--------------------#
  
  # reduction factor in r-direction
  r_skip = 2
  
  # reduction factor in z-direction
  z_skip = 4

  # ---------------------------- computed equation coefficients ---------------------------

  # determine proper timestep:
  if(n_c==0):
    dt = np.min( (0.1, t_i/5) )
  elif(t_i==0):
    dt = np.min( (0.1,0.01*D*t_p/5) )
  else:
    dt = np.min( (0.1,0.01*D*t_p/5,t_i/5) )
 
  # determine number of integration steps at each stage:
  N_i = np.around(t_i/dt)
  if(n_c==0):
    N_p=0
  else:
    N_p = np.around(t_p/dt)
    
  N_c = np.around(t_c/dt)
  
  # total number of integration steps
  N = N_i+n_c*N_p+N_c
  
  # total simulation duration
  T = dt*N

  # build input vector, which catalogs the HIFU beam on/off operation 
  if (N_p != 0):
    pulse = np.zeros( (1,N_p), dtype=np.float64 )
    for n in np.arange( 0, np.around(0.01*D*N_p)-1 ):
      pulse[n] = 1
    pulses = pulse
    for m in np.arange(2,n_c):
      pulses = np.vstack((pulses,pulse))
    
  sequence = np.zeros((1,N), dtype=np.float64)
  
  if (N_c != 0):
    cooloff = np.zeros( (1,N_c), dtype=np.float64 )
  
  if(N_i != 0):
    initial = np.ones( (1,N_i) , dtype=np.float64)

  for n in np.arange(0,N_i):
    sequence[0,n] = initial[0,n]

  if N_p > 0:
    for n in np.arange(N_i,N_i+n_c*N_p-1):
      sequence[0,n] = pulses[0,n-N_i]
 
  for n in np.arange(N_i+n_c*N_p,N-1):
    sequence[0,n] = cooloff[0,n-N_i-n_c*N_p]

  # build grid vectors:
  #--------------------#

  # time nodes
  t = np.linspace(0,T,num=N+1,endpoint=True, retstep=False)
  
  #print dt, T, N+1
               
  # size of J,M based on acoustic models
  J, M = np.size(H,axis=0), np.size(H,axis=1)   
 
  # rescaled radial
  r_bht = r[ 0 : np.round(J/2) : r_skip, 0]

  # rescaled axial
  #print np.shape( np.shape(z) ), np.shape(z), np.shape( np.shape( np.squeeze(z) ) ), np.shape( np.squeeze(z) ), np.int( np.shape( np.shape( np.squeeze(z) ) )[0] )
  if ( np.shape(z)[0] > np.shape(z)[1] ):
    z_bht = z[ 0 : M+1 : z_skip, 0]
  else:
    z_bht = z[ 0, 0 : M+1 : z_skip]
                    
  # rescaled heating rate
  H_bht = H[ 0 : np.round(J/2) : r_skip, 0 : M-1 : z_skip ]
  
  # reassign J,M
  J_bht, M_bht = np.size(H_bht, axis=0), np.size(H_bht,axis=1)

  # print output
  print '\n\tdt = %2.2f sec\tN = %d' %(dt,N)
  print '\tdr = %2.2f mm\tJ = %d' %(10*r_bht[1],J_bht)
  print '\tdz = %2.2f mm\tM = %d' %(10*z_bht[1],M_bht)
  
  return C1,C2,k1,k2,w1,w2,N,t,sequence,dt,J_bht,M_bht,z_bht,r_bht,H_bht,T0
