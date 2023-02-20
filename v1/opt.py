



def equivalent_time(T, N):
  
  """
  
  Computes the integrand of the CEM43 thermal dose integral for an N-vector of temperatures T.
  
  """
  
  import numpy as np

  T_inf = 37.0

  diff = 43.0 - T_inf

  for n in np.arange( 0, N-1 ):
    if ( T[n] <= diff ):
      Tcem = np.power( 0.25, (diff-T[n]) )
    else:
      Tcem = np.power( 0.50, (diff-T[n]) )

  del T_inf, diff

  return Tcem
  
def timing(p1,p2,t_start,z,d,n):
  
  """"
  
  Time keeping routine, formats output to screen.
  
  """
  
  from numpy import floor
  from time import time
  
  if ( p2 > p1 ):
    times = time() - t_start 
    h    = floor(times/3600) 
    times = times - h*3600
    m    = floor(times/60)
    times = times - m*60
    p1   = p2
    
    print '\t%3.1f\t\t%02d:%02d:%04.1f\t\t%03d' %(z*d, h, m, times, n)
    
    del times, h, m

    return p1

def KZK_parameters(drive,efficiency):
  
  """
  
  Consists of user-defined input parameter definitions, including material and transducer 
  parameters, as well as the size of the computational domain; returns required computed 
  parameters for integration of the acoustic and thermal model equations.
  
  """
  
  import numpy as np
  from math import pi, log

#---------------------------- User-defined input parameters -----------------------------

  # material 1 parameters: WATER
  #-----------------------------
  
  # small-signal sound speed [m/s]
  c1      = 1482.0 
  
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

  # material 2 parameters: LIVER
  #-----------------------------
  
  # small-signal sound speed [m/s]
  c2      = 1570.0 
  
  # mass density [kg/m^3]
  rho2    = 1070.0 
  
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
  
  # power [W]
  P   = drive*efficiency 

  # computational domain size
  #-----------------------------
  
  # max radius [cm]
  R = a 
  
  # max axial distance [cm]
  Z = 1.5*d 
  
  # propation distance through tissue [cm]
  ztissue = 1.3
  
  # material transition distance [cm]
  z_ = d - ztissue 

  # number of harmonics included in simulation
  K = 48 

  # ---------------------------- computed equation coefficients ---------------------------

  # peak pressure at transducer face
  p0 = np.sqrt( 2.0*rho1*c1*P/pi/( (a/100.0)**2 - (b/100.0)**2 ) )   
  
  # nonlinear coefficient
  N1 = 2.0*pi*p0*beta1*(d/100.0)*f/rho1/( c1**3 )           
  N2 = 2.0*pi*p0*beta2*(d/100.0)*f/rho2/( c2**3 )
  
  # linear pressure gain
  G1 = pi*f/c1/(d/100.0)*( (a/100.0)**2 )                
  G2 = pi*f/c2/(d/100.0)*( (a/100.0)**2 )

  # attenuation coefficients
  gamma1 = np.zeros((K,1))                               
  gamma2 = np.zeros((K,1))

  # scaled frequency
  h = f*np.arange(1,K+1)/1e6                                  

  # convert alpha from dB/m to Np
  alpha01 = (d/100.0)*alpha01/8.686                   
  alpha02 = (d/100.0)*alpha02/8.686

  # define absorption as power law when exponent is/isn't an integer
  if(eta1==1):                                                                                      
    gamma1 = alpha01*h**(1.0 - 1j*2.0*log(h)/pi )
  elif(eta1==2):
    gamma1 = alpha01*h**2.0
  else:
    gamma1 = alpha01*h**eta1 - 1j*2.0*(alpha01*h**eta1 - alpha01*h)/(eta1 - 1.0)/pi
  if(eta2==1):
    gamma2 = alpha02*h**(1.0 - 1j*2.0*log(h)/pi)
  elif(eta2==2):
    gamma2 = alpha02*h**2.0
  else:
    gamma2 = alpha02*h**eta2 - 1j*2.0*(alpha02*h**eta2 - alpha02*h)/(eta2 - 1.0)/pi
    
  # absorption coefficients
  alpha1 = np.zeros( (K,1) )                               
  alpha2 = np.zeros( (K,1) )
  theta01 = (d/100.0)*theta01/8.686
  theta02 = (d/100.0)*theta02/8.686
  
  # define absorption as power law when exponent is/isn't an integer
  if(phi1==1):                                                                                      
    alpha1 = theta01*h**(1.0 - 1j*2.0*log(h)/pi )
  elif(phi1==2):
    alpha1 = theta01*h**2.0
  else:
    alpha1 = theta01*h**phi1 - 1j*2.0*(theta01*h**phi1 - theta01*h)/(phi1 - 1.0)/pi
  if(phi2==1):
    alpha2 = theta02*h**(1.0 - 1j*2.0*log(h)/pi)
  elif(phi2==2):
    alpha2 = theta02*h**2.0
  else:
    alpha2 = theta02*h**phi1 - 1j*2.0*(theta02*h**phi2 - theta02*h)/(phi2 - 1.0)/pi

  # nondimensionalize grid dimensions:
  R = R/a
  Z = Z/d

  del h
    
  return p0,c1,c2,rho1,rho2,N1,N2,G1,G2,gamma1,gamma2,alpha1,alpha2,a,b,d,f,R,Z,z_,K
  
 def computational_grid(Z,R,G,a,d,gamma,N):

  """

  Generates node vectors for discretization in axial and radial directions

  """
  
  from math import pi
  import numpy as np

  # est. points per wavelength in axial direction
  ppw_ax = 40

  # number of meshpoints in axial direction 
  M = np.around(ppw_ax*Z*G)

  # axial stepsize
  dz = Z/(M-1.0)
	  
  # axial node vector
  z = np.linspace(0, Z, num=np.int(M))    

  # est. points per wavelength in radial direction
  ppw_rad = 50

  # number of meshpoints in radial direction   
  J = np.around(ppw_rad*R*G/pi) 

  # [0,R] is physical, the extra 0.25 is for PML
  R_ext = R+0.25 
  dr = R_ext/(J-1.0)

  # radial node vector
  r = np.linspace(0.0, R_ext, num=np.int(J))
    
  # number of nodes in [0,R]
  J_ = np.ceil( J*R/R_ext )

  # print parameters
  print ('\tdr = %2.3f mm\tJ = %d') %(10.0*a*dr, J)
  print ('\tdz = %2.3f mm\tM = %d\n') %(10.0*d*dz, M)

  del ppw_ax, ppw_rad, R_ext
  
  return M,J,J_,dz,dr,z,r
  
def initial_condition(J,K,G,r,ir,limit):

  """
  
  Determines the initial condition for the KZK equation.  In this case, it's a uniform pressure 
  distribution with a phase shift corresponding to a quadratic approximation of a spherical 
  converging wave, appropriate for above f/1.37. 
  
  """  
  
  import numpy as np
  
  v = np.zeros( (2*J,K), dtype=np.float64)
  
  # j runs from 0 to J-1, so can be used to populate an array with J values.
  for j in np.arange(0,J):

    if ( (np.abs(r[j]) >= ir) and (np.abs( r[j] ) <= limit) ):

      arg      = G*r[j]*r[j]
      v[j,0]   = np.cos(arg) 
      v[J+j,0] =-np.sin(arg) 
      
  del j,arg

  return v
  
def KZK_operators(r,R,G,dz,dr,J,k,gamma):

  """
  
  Computes second-order diagonally implicit Runge-Kutta and Crank-Nicolson operators for 
  integrating the k-th harmonic equation.
  
  """
  
  import numpy as np
  import os
  from scipy.io import loadmat, savemat
  from scipy.sparse import spdiags, eye, bmat
  from scipy.sparse.linalg import splu, factorized
  from math import pi
  
  iverbose = 0
  
  # PML vector
  w = np.zeros( (J,1) )
  for j in np.arange(0,J):
    if ( r[j] > R ): 
      w[j] = 16.0*( r[j] - R )**2
 
  w = np.squeeze( np.exp( 1j*pi*w/2.0 ) )
  
  # python-specific shift of k to match matlab, as iterator in loop is from 0:K, i.e. 0 to K-1 incl. 
  k = k+1
  
  supdiag  = np.squeeze(np.zeros((J,1)))
  diagonal = np.squeeze(np.zeros((J,1)))
  subdiag  = np.squeeze(np.zeros((J,1)))

  # build finite-difference matrix:
  #--------------------------------
  
  # copy to dummy variable
  rr = np.copy(r)
  
  # prevent divide by zero error
  rr[0] = 1.0
  
  supdiag  = np.conjugate( ( w/dr + 0.5/rr )/dr )
  
  diagonal = np.conjugate( -2.0*w/dr/dr ) 
  
  subdiag  = np.conjugate( ( w/dr - 0.5/rr )/dr )
    
  temp = np.array( [supdiag, diagonal, subdiag] )
  
  A = spdiags(temp, np.array([-1,0,1]), J, J)
    
  A = A.transpose()

  A = A.conjugate()

  A[0,1] = -A[0,0]

  # split into real and imaginary parts:
  ReA = A.real
  ReA = 0.25*ReA/G/k
  
  ImA = A.imag
  ImA = 0.25*ImA/G/k

  alpha = np.real(gamma)*eye(J, J, dtype=np.float64)
  beta  = np.imag(gamma)*eye(J, J, dtype=np.float64)
  
  alpha = alpha.tocsc()
  beta  = beta.tocsc()
  
  # build real-valued block system:
  M = bmat( [ [ImA-alpha, -ReA-beta], [ReA+beta, ImA-alpha] ] )
  
  I = eye(2*J, 2*J, dtype=np.float64)
  
  # DIRK2 matrices:
  IRK1 = I - dz*( 1.0 - 1.0/np.sqrt(2.0) )*M
  IRK2 = M
  
  # specify sparse matrix storage type
  IRK1 = IRK1.tocsc()
  IRK2 = IRK2.tocsc()
  
  #IRK1 = splu(IRK1)
  #IRK2 = splu(IRK2)
  
  IRK1 = factorized( IRK1 )
  
  # CN matrices:
  CN1 = I - 0.5*dz*M
  CN2 = I + 0.5*dz*M
  
  # specify sparse matrix storage type
  CN1 = CN1.tocsc()
  CN2 = CN2.tocsc()
    
  del I,ReA,ImA,alpha,beta,M,A,supdiag,diagonal,subdiag,w,j,rr
  
  return IRK1,IRK2,CN1,CN2

    
    
def TDNL(u,U,X,K,J,c,cutoff,Ppos,Pneg,I_td):

  """
  
  Converts spectrum to one cycle of the time-domain waveform and integrates the invicid Burger's 
  equation using upwind/downwind method with periodic boundary conditions. 
  TDNL stands for Time Domain NonLinear.
  
  """

  import numpy as np
  import warnings
  from numpy.fft import fft, ifft
    
  warnings.simplefilter("ignore", np.ComplexWarning)
    
  # linear case - do nothing
  if (K==1):

    print "linear!"
  
  # nonlinear case - enter loop
  else:
    
    # starting at J-1, to 0, in J steps, inclusive. 
    # Note this is one of the differences between arange and linspace. 
    # As an array index j is shifted
    for j in np.linspace(J-1, 0, num=J):
      
      # execute nonlinear step only if amplitude is above cutoff; row j=1 is always computed 
      # so plots look nice and smooth
      I_td[j] = 0
      
      if ( (np.sqrt(u[j,0]**2+u[J+j,0]**2) > cutoff) or (j==1) ):
	
	# convert U from sin/cos representation to ***complex*** exponential 
	U[ 0, 0 ] = 0.0 #+ 0j
	
	# runs from index 1 to K inclusive, ie K-1 values. 
	# Matlab 2:K+1, so python indices 1 to K inclusive
	U[ 0, 1 : K+1 ] = np.real(u[j,:]) - 1j*np.real(u[j+J,:])
	
	# runs from index 2*K+1 to K inclusive
	# Matlab 2K:K+2, so python indices 2K-1 to K+1 inclusive
	U[ 0, 2*K-1 : K : -1 ] = np.real(u[j,0:K-1]) + 1j*np.real(u[j+J,0:K-1])
	
	# transform to time domain, and convert U to ***real*** number
	U = K*np.real( ifft(U) )

	# time-domain intensity
	I_td[j] = np.trapz( np.squeeze(U**2) )

	# determine how many steps necessary for CFL<1 ( CFL<0.8 to be safe )
	cfl = 0.8
	P = np.ceil( c*np.max(np.squeeze(U))/cfl )
	
	# Nonlinear integration (upwind/downwind) algorithm. Note that p runs from 0 to P-1, 
	# which is P steps in total, as calculated by the CFL condition
	for p in np.arange(0,P):
	  
	  # for each frequency component
	  for k in np.arange( 0, 2*K ):
	  
	    if ( U[0,k]<0.0 ):
	    
	      if (k==0):
		X[0,k] = U[0,k] + c*( U[0,0]*U[0,0] - U[0,2*K-1]*U[0,2*K-1] )/P/2.0
	      else:
		X[0,k] = U[0,k] + c*( U[0,k]*U[0,k] - U[0,k-1]*U[0,k-1] )/P/2.0

	    else:
	      
	      if ( k==(2*K-1) ):
		X[0,k] = U[0,k] + c*( U[0,0]*U[0,0] - U[0,k]*U[0,k] )/P/2.0
	      else:
		X[0,k] = U[0,k] + c*( U[0,k+1]*U[0,k+1] - U[0,k]*U[0,k] )/P/2.0

	  # update output argument U 
	  U = X

	# update time-domain intensity
	I_td[j] = I_td[j] - np.trapz( np.squeeze( np.real(X) )**2 ) 
	
	# update peak positive pressure
	Ppos[j] = np.max( np.real(X) )
	
	# update peak negative pressure
	Pneg[j] = np.min( np.real(X) )
	
	# transform back to frequency domain: at this point X takes ***complex*** values again
	X = fft(X)/K
	
	# convert back to sin/cos representation:
	u[j,:]   = np.real( X[0,1:K+1] )
	u[j+J,:] =-np.imag( X[0,1:K+1] )
	
  return u,U,Ppos,Pneg,I_td 


def axisymmetricKZK(drive,efficiency):

  """
  
  Driver for axisymmetric KZK integrator.  
  
  """
  
  import numpy as np
  import matplotlib.pyplot as plt
  import matplotlib.mlab as mlab
  import time, os, warnings
  
  from math import pi
  from scipy.io import loadmat, savemat
  from scipy.sparse.linalg import spsolve, factorized
  from scipy.linalg import solve, lu
  
  import equivalent_time, timing, KZK_parameters, computational_grid, initial_condition, KZK_operators, TDNL
  
  # get system parameters
  p0,c1,c2,rho1,rho2,N1,N2,G1,G2,gamma1,gamma2,alpha1,alpha2,a,b,d,f,R,Z,z_,K = KZK_parameters.KZK_parameters(drive,efficiency)

  # set size of arrays which store complex values for the K harmonics
  K2 = 2*K

  # print coefficients to screen
  print '\n\tp0 = %2.2f MPa' %(1e-6*p0)
  print '\tN1 = %1.2f\tN2 = %1.2f' %(N1,N2)
  print '\tG1 = %3.2f\tG2 = %3.2f\n' %(G1,G2)

  # check F-number satisfies paraxial criteria
  F = 0.5*d/a
  if (F<1.37):
    print ' *** Warning -- f/%1.2f exceeds the conditions under which KZK is derived (> f/1.37) *** \n' %(F)

  # grid set-up
  M,J,J_,dz,dr,z,r = computational_grid.computational_grid(Z,R,max(G1,G2),a,d,gamma2[0],N2)
    
  # dependent variables u, v
  u = np.zeros((2*J,K))
  
  limit = 1.0/np.sqrt( 1.0 - (a/d)**2 )
  v = initial_condition.initial_condition(J,K,G1,r,b*limit/a,limit)
  v[0:J,0]   = v[0:J,0]*np.sqrt( 1.0-(r/d)**2 )  
  v[J:2*J,0] = v[J:2*J,0]*np.sqrt( 1.0 - (r/d)**2 ) # why this is done over two lines i don't know!
  
  op = [KZK_operators.KZK_operators(r,R,G1,dz,dr,J,k,gamma1[k]) for k in np.arange(0,K)] 
  IRK11, IRK12, CN11, CN12 = zip(*op)
  
  op = [KZK_operators.KZK_operators(r,R,G1,dz,dr,J,k,gamma2[k]) for k in np.arange(0,K)] 
  IRK21, IRK22, CN21, CN22 = zip(*op)
       
  # IRK slope vectors
  k1 = np.zeros( (2*J,1) )
  k2 = np.zeros( (2*J,1) )

  # IRK coefficients
  b1 = 1.0/np.sqrt(2.0)
  b2 = 1.0 - b1

  # parameters for nonlinear integration
  #-------------------------------------
  
  # nonlinear term integration parameters
  mu1 = N1*K*dz/pi         
  mu2 = N2*K*dz/pi
  
  # cutoffs for performing nonlinear integration
  cutoff1 = gamma1[0]/10.0/N1
  cutoff2 = gamma2[0]/10.0/N2
  
  # data vectors
  X = np.zeros((1,K2)) #, dtype=complex)
  Y = np.zeros((1,K2), dtype=complex)
  
  #X = np.zeros((1,K2))
  #Y = np.zeros((1,K2))
  
  # waveform at position of maximum intensity 
  Xpeak = np.zeros( (1,K2) )

  # for plotting routines:
  #-------------------------------------
  
  # Heating rate matrices
  H  = np.zeros((J_,M+1))
  H2 = np.zeros((J_,M+1))
  H[:,0] = np.real( gamma1[0] )*( v[0:J_,0]**2 + v[J:J+J_,0]**2 )
  I = np.zeros((J_,M+1))
  I[:,0] = v[0:J_,0]**2 + v[J:J+J_,0]**2
  I_td = np.zeros((J,1))
  dt = 1.0/f/(K2-1.0)
  
  # axial intensity vector
  Ix = np.zeros((M+1,1))
  Ix[0] = v[0,0]**2 + v[J,0]**2
  
  # radial intensity vector
  Ir = np.zeros((J_,1))

  # first (up to) 5 harmonic pressure amplitudes in axial direction
  if (K<5): 
    kk = K 
  else:
    kk = 5
  p5x = np.zeros((kk,M+1))
  p5x[0,0] = np.sqrt( v[0,0]*v[0,0] + v[J,0]*v[J,0] )

  # first (up to) 5 harmonic pressure amplitudes in radial direction
  p5r = np.zeros((kk,J_))

  # axial peak positive pressures
  peak = np.zeros((M+1,1))

  # axial peak negative pressures
  trough = np.zeros((M+1,1))
  peak[0]   = 1.0
  trough[0] =-1.0

  # for monitoring simulation: 
  #-------------------------------------
  
  # amplitude of Kth harmonic
  amp_K = 0

  # for determining peak pressure waveform and location of its occurence:
  #-------------------------------------
  
  # index of last meshpoint in material 1? ceil rounds up
  m_t = np.ceil(z_/dz/d)
  if(m_t>M): 
    m_t = M 

  # index of meshpoint nearest focus   
  m_f = np.around(M/Z)

  # peak axial pressure
  p_peak = 0.0

  # distance at which peak axial pressure occurs        
  z_peak = 0.0 

  # allocate memory
  Ppos = np.zeros((J,M)) 
  Pneg = np.zeros((J,M))

  # integrate the equations through material 1
  print 'Integrating KZK equation...\n'
  print '\tz (cm)\t\ttime (hr:min:sec) \tn\n'

  # start timer
  steps = np.floor( np.linspace(0,M,num=10,endpoint=False) )
  steps = steps.astype(int)
  t_start = time.time()
  p1 = 0
    
  #--------------------------------------------------------------------------------------------------

  # spatial front: from 0 to m_t-1
  for m in np.arange(0,m_t):
    
    # compute updated v
    v,X,Ppos[:,m],Pneg[:,m],I_td = TDNL.TDNL(v,X,Y,K,J,mu1,cutoff1,Ppos[:,m],Pneg[:,m],I_td)
    
    # update time-domain intensity
    I_td = f*dt*I_td/dz
    
    # update peak positive pressure
    peak[m+1]   = np.max(np.real(X))
    
    # update peak negative pressure
    trough[m+1] = np.min(np.real(X))
    
    # if at focus get radial information
    if (m==m_f): 
      # radial intensity as sum of harmonics
      if (K>1):
	Ir = np.sum( np.transpose( v[0:J_,:] )**2 )
	Ir = Ir + np.sum( np.transpose( v[J:J+J_,:] )**2 )
      else:
	Ir = v[0:J_]**2 + v[J:J+J_]**2
      # first five harmonics
      p5r = np.sqrt( v[0:J_,0:kk]**2 + v[J:J+J_,0:kk]**2 )

    # peak is maximal get updated data
    if (peak[m+1]>p_peak):
      p_peak = peak[m+1]
      z_peak = z[m+1]
      Xpeak  = X
    
    # if in the near field solve with an implicit Runge-Kutta scheme, else using a Crank-Nicolson scheme
    if (z[m]<0.3):
      
      # for each harmonic
      for k in np.arange(0,K):

	#k1 = spsolve(IRK11[k], IRK12[k]*v[:,k] )
	k1 = IRK11[k]( IRK12[k]*v[:,k] )
	#k2 = spsolve(IRK11[k], IRK22[k]*(v[:,k] + dz*b1*k1) )
	k2 = IRK11[k](IRK22[k]*(v[:,k] + dz*b1*k1) )
	
	# update acoustic field
	u[:,k] = v[:,k] + dz*( b1*k1 + b2*k2 ) 

    else:
      for k in np.arange(0,K):
	u[:,k] = spsolve(CN11[k], CN12[k]*v[:,k] )

    # reassign working variable
    v = u
    
    # reassign amplitude of K-th harmonic
    pl = np.sqrt( v[0,K-1]**2 + v[J,K-1]**2 )
    if (pl > amp_K): 
      amp_K = pl

    # update heating rates
    for j in np.arange(0,J_):
      H[j,m+1]  = np.sum( np.real( np.transpose(gamma1[:]) )*( u[j,:]**2 + u[J+j,:]**2 ) )
      H2[j,m+1] = I_td[j]
      I[j,m+1]  = np.sum( u[j,:]**2 + u[J+j,:]**2 )
    
    # update axial Intensity
    Ix[m+1] = np.sum( v[0,:]**2 + v[J+1,:]**2 )
    
    # gain check
    if ( Ix[m+1] > 2*G1*G1 ):
      print '\tStopped - computation became unstable at z = %2.1f cm.\n' %(d*z[m])
      r = a*r[0:J_]
      z = d*z
      #KZK_radial_plots.KZK_radial_plots(r,Ir,H(:,round((M+1)/Z)),p5r,p0,rho2,c2,R,a); 
      #KZK_axial_plots.KZK_axial_plots(z,Ix,p5x,H(1,:),peak,trough,p0,rho1,rho2,c1,c2,d,Z,a,m_t)
      exit(0)
    
    # axial pressure values for first five harmonics
    p5x[:,m+1] = np.sqrt( v[0,0:kk]*v[0,0:kk] + v[J,0:kk]*v[J,0:kk] )
    
    # timing routine
    if m in steps:
      p2 = np.floor(10*(m+1)/M)
      p1 = timing.timing(p1,p2,t_start,z[m],d,m)
    
  # material 2:
  #------------------------------------------------------------------------------------------------
  
  # impediance mismatch loss
  v = 2.0*rho2*c2*v/(rho1*c1+rho2*c2)
  
  for m in np.arange(m_t,M):
    v,X,Ppos[:,m],Pneg[:,m],I_td = TDNL.TDNL(v,X,Y,K,J,mu2,cutoff2,Ppos[:,m],Pneg[:,m],I_td)
    I_td = f*dt*I_td/dz
    peak[m+1]   = np.max(np.real(X))
    trough[m+1] = np.min(np.real(X))
    
    if (m==m_f):
      if(K>1):
	Ir = sum( ( np.transpose(v[0:J_,:]) )**2 )
	Ir = Ir + sum( ( np.transpose(v[J:J+J_,:]))**2)
      else:
	Ir = v[0:J_]**2 + v[J:J+J_]**2
      
      p5r = np.sqrt( v[0:J_,0:kk]**2 + v[J:J+J_,0:kk]**2 )
  
    if (peak[m+1]>p_peak):
      p_peak = peak[m+1]
      z_peak = z[m+1]
      Xpeak = X
    
    if (z[m]<0.3):

      if (m==m_t):
	
	for k in np.arange(0,K):
	  #k1 = spsolve(IRK11[k], IRK12[k]*v[:,k] )
	  #k2 = spsolve(IRK21[k], IRK22[k]*(v[:,k] + dz*b1*k1) ) 
	  
	  #k1 = spsolve(IRK11[k], IRK12[k]*v[:,k] )
	  k1 = IRK11[k]( IRK12[k]*v[:,k] )
	  #k2 = spsolve(IRK11[k], IRK22[k]*(v[:,k] + dz*b1*k1) )
	  k2 = IRK21[k](IRK22[k]*(v[:,k] + dz*b1*k1) )
	  
	  u[:,k] = v[:,k] + dz*(b1*k1 + b2*k2)

	u = (1.0 - (rho1*c1-rho2*c2)/(rho1*c1+rho2*c2))*u

      else:
	
	for k in np.arange(0,K):
	  #k1 = spsolve(IRK21[k], IRK22[k]*v[:,k]) 
	  k1 = IRK21[k]( IRK22[k]*v[:,k] ) 
	  #k2 = spsolve(IRK21[k], IRK22[k]*(v[:,k]+dz*b1*k1)) 
	  k2 = IRK21[k](IRK22[k]*(v[:,k]+dz*b1*k1))
	  u[:,k] = v[:,k] + dz*(b1*k1 + b2*k2)

    else:
      
      for k in np.arange(0,K):
	u[:,k] = spsolve(CN21[k], CN22[k]*v[:,k]) 

    v = u
    pl = np.sqrt( v[0,K-1]**2 + v[J,K-1]**2)
    
    if (pl > amp_K):
      amp_K = pl
    
    for j in np.arange(0,J_):
      H[j,m+1]  = sum( np.real( np.transpose(gamma2[:]) )*( u[j,:]**2 + u[J+j,:]**2) )
      H2[j,m+1] = I_td[j]
      I[j,m+1]  = sum( u[j,:]**2 + u[J+j,:]**2 )

    Ix[m+1] = sum( v[0,:]**2 + v[J,:]**2 )
    
    if ( Ix[m+1] > 2*G2*G2 ):
      print '\tStopped - computation became unstable at z = %2.1f cm.\n' %(d*z[m])
      r = a*r[0:J_]
      z = d*z
      #KZK_radial_plots.KZK_radial_plots(r,Ir,H(:,round((M+1)/Z)),p5r,p0,rho2,c2,R,a); 
      #KZK_axial_plots.KZK_axial_plots(z,Ix,p5x,H[1,:],peak,trough,p0,rho1,rho2,c1,c2,d,Z,a,m_t)
      exit(0)
    
    p5x[:,m+1] = np.sqrt( v[0,0:kk]*v[0,0:kk] + v[J+1,0:kk]*v[J+1,0:kk] )
    
    if m in steps:
      p2 = np.floor(10*(m+1)/M)
      p1 = timing.timing(p1,p2,t_start,z[m],d,m)
    
  # dimensionalize H
  H[:,0:m_t+1]  = 1e-4*p0*p0*H[:,0:m_t+1]/rho1/c1/d
  H[:,m_t+1:M]  = 1e-4*p0*p0*H[:,m_t+1:M]/rho2/c2/d

  # dimensionalize H2
  H2[:,0:m_t+1] = 1e-4*0.5*p0*p0*H2[:,0:m_t+1]/rho1/c1/d
  H2[:,m_t+1:M] = 1e-4*0.5*p0*p0*H2[:,m_t+1:M]/rho2/c2/d

  # dimensionalize I
  I[:,0:m_t+1] = 1e-4*0.5*p0*p0*I[:,0:m_t+1]/rho1/c1   
  I[:,m_t+1:M] = 1e-4*0.5*p0*p0*I[:,m_t+1:M]/rho2/c2

  H = np.real(H + H2)

  print '\n\tmax(|p_K|/p0) = %e\n' %(amp_K)
  print '\n\tratio 2nd harmonic to 1st   = %6.5f \n' %( max( np.max(p5x[1,:]), np.max(p5r[:,1]))/max(np.max(p5x[0,:]), np.max(p5r[:,0])) )
  print '\tratio 3rd harmonic to 1st   = %6.5f \n' %( max(np.max(p5x[2,:]),np.max(p5r[:,2]))/max(np.max(p5x[0,:]),np.max(p5r[:,0])) )
  print '\tratio 1st harmonic to total = %6.5f \n' %( max(np.max(p5x[0,:]),np.max(p5r[:,0]))/np.max(peak) )
  print '\tratio 2nd harmonic to total = %6.5f \n' %( max(np.max(p5x[1,:]),np.max(p5r[:,1]))/np.max(peak) )

  # rescale r and chop so that PML region is excluded in plots
  r = a*r[0:J_]

  # rescale z 
  z = d*z

  # peak positive 
  Ppos = Ppos[0:J_,:]

  # peak negative
  Pneg = Pneg[0:J_,:]

  del H2,amp_K
    
  return z,r,H,I,Ppos,Pneg,Ix,Ir,p0,p5r,p5x,peak,trough,rho1,rho2,c1,c2,R,d,Z,M,a,m_t,f,Xpeak,z_peak,K2,z_
  
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
               
  # size of J,M based on acoustic models
  J,M = np.size(H,axis=0), np.size(H,axis=1)   
 
  # rescaled radial
  r_bht = r[ 0 : np.round(J/2) : r_skip, 0]

  # rescaled axial
  z_bht = z[0, 0 : M+1 : z_skip ]
                    
  # rescaled heating rate
  H_bht = H[ 0 : np.round(J/2) : r_skip, 0 : M-1 : z_skip ]
  
  # reassign J,M
  J_bht, M_bht = np.size(H_bht, axis=0), np.size(H_bht,axis=1)

  # print output
  print '\n\tdt = %2.2f sec\tN = %d' %(dt,N)
  print '\tdr = %2.2f mm\tJ = %d' %(10*r_bht[1],J_bht)
  print '\tdz = %2.2f mm\tM = %d' %(10*z_bht[1],M_bht)
  
  return C1,C2,k1,k2,w1,w2,N,t,sequence,dt,J_bht,M_bht,z_bht,r_bht,H_bht,T0  
  
def BHT_operators(z,r,K1,K2,P1,P2,dt,zz):
  
  """
  
  builds operators for IRK2 integration, based on r_bht, z_bht
  
  """ 
  
  import numpy as np
  from math import pi, log
  from scipy import transpose
  from scipy.sparse import spdiags, eye
  
  J = np.size(r, axis=0)
  M = np.size(z, axis=0)
 
  dr = r[1]
  dz = z[1]
  
  JM = J*M
  m_ = np.around(zz*M)

  # build matrix operator's vector "bands" 
  alpha_plus  = np.squeeze( np.ones( (JM,1), dtype=np.float64 )/dz/dz )
  alpha_minus = np.squeeze( np.ones( (JM,1), dtype=np.float64 )/dz/dz )
  
  bp = np.squeeze( np.zeros( (J,1), dtype=np.float64 ))
  bm = np.squeeze( np.zeros( (J,1), dtype=np.float64 ))
  
  bp[0] = 2.0/dr/dr
  bp[1:J] = (1.0/dr + 0.5/r[1:J] )/dr
  bm[1:J] = (1.0/dr - 0.5/r[1:J] )/dr
  
  beta_plus  = np.squeeze( np.zeros( (JM,1), dtype=np.float64) )
  beta_minus = np.squeeze( np.zeros( (JM,1), dtype=np.float64) )
  
  # jm runs from 0 to JM-1
  for jm in np.arange(0,JM):
    
    if (np.mod(jm+1,M)==0): 
      alpha_plus[jm] = 0.0   
    
    if (np.mod(jm+1,M)==1): 
      alpha_minus[jm] = 0.0
    
    # number of times iterate is a multiple of M
    n = np.ceil( (jm+0)/M )
    beta_plus[jm]  = bp[n]
    beta_minus[jm] = bm[n]
  
  gamma = np.squeeze( -2.0*(1.0/dr/dr + 1.0/dz/dz)*np.ones( (JM,1), dtype=np.float64) )

  # build diffusivity coefficient matrix
  k1 = K1*np.ones( (m_,1), dtype=np.float64)
  k2 = K2*np.ones( (M-m_,1), dtype=np.float64)
  k = np.vstack((k1,k2))
  bigk = k
  # j runs from 0 to J-2, i.e. in total there are J-1 items
  for j in np.arange(0,J-1):
    bigk = np.vstack((bigk,k))
  
  K = spdiags(np.squeeze(bigk), 0, JM, JM );

  # build perfusion coefficeint matrix:
  p1 = P1*np.ones( (m_,1), dtype=np.float64 )
  p2 = P2*np.ones( (M-m_,1), dtype=np.float64 )
  p  = np.vstack((p1,p2))
  bigp = p
  for j in np.arange(0,J-1):
    bigp = np.vstack((bigp,p))
  
  P = spdiags(np.squeeze(bigp), 0, JM, JM)
  
  temp = np.array([np.squeeze(beta_plus),np.squeeze(alpha_plus),np.squeeze(gamma),np.squeeze(alpha_minus),np.squeeze(beta_minus)], dtype=np.float64 )
  
  # put it all together
  A = spdiags( temp, np.array([-M, -1, 0, 1, M], dtype=np.float64 ), JM, JM )
  A = A.T
  A = A*K - P
  B = eye(JM,JM) - 0.5*dt*A

  del alpha_plus,alpha_minus,bp,bm,beta_plus,beta_minus,gamma,k1,k2,k,bigk,p1,p2,p,bigp,P,temp
  
  return A,B
  
  
  
  
  #--------------------------------------------------------------------------------------------------
    
  def axisymmetricBHT(H, z, r):

  """

  Driver for axisymmetric Bioheat equation, solved using a diagonally dominiant implicit 
  Runge-Kutta scheme

  """

  #--------------------------------------------------------------------------------------------------
  # import packages

  import numpy as np
  import os, time, datetime, warnings

  from math import pi, log

  from scipy.sparse.linalg import spsolve, factorized, splu
  from scipy.linalg import solve, lu

  import matplotlib.pyplot as plt
  import matplotlib.mlab as mlab

  from scipy.io import loadmat, savemat

  import equivalent_time, timing, KZK_parameters, BHT_parameters, BHT_operators, axisymmetricKZK

  #--------------------------------------------------------------------------------------------------
  ## preamble, warnings and information, clear screen

  ## clear screen
  #os.system('cls' if os.name == 'nt' else 'clear')
  
  from scipy.sparse import SparseEfficiencyWarning
  warnings.simplefilter("ignore", SparseEfficiencyWarning)

  #--------------------------------------------------------------------------------------------------

  # set level of verbose output
  iverbose=0
  
  # set degree of data loading 
  iload = 0
  
  # specify whether figure is rendered
  ifig = 0
  
  # define whether data is loaded
  manual=1
  
  # set whether LU decomposition is applied to spead up solution method.
  ilu=1

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

  # set location of filename
  filename=os.path.expanduser('~/Desktop/Thermal_pycomp_5s_-12dbm_1.mat')

  # check whether file exists
  if os.path.isfile(filename):
    # load canonical data set, but only load what is strictly necessary if iload=0, else load all.
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
      # load minimum ammount
      data = loadmat(filename, variable_names=['z','r','H'] )
      z = data['z']
      r = data['r']
      H = data['H']
  else:
    print "No file to load in this instance!"
    
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
  if manual==1:
    A,B = BHT_operators.BHT_operators(z_bht,r_bht,D1,D2,P1,P2,dt,z_/d/Z)
    if ilu==1:
      solve1 = factorized(A)
      solve2 = factorized(B)
    if ilu==2:
      B = B.tocsc()
      Binv = splu(B)

  # index at material interface 
  m_ = np.around(z_*M_bht/d/Z)
  if (m_>M_bht): 
    m_ = M_bht 
    
  # print interface location and time-step
  if (iverbose==1):
    print m_, dt
    
  # rescale H_bht to degrees/second
  H_bht[:,0:m_-1] = 1e6*H_bht[:,0:m_-1]/C1/rho1
  H_bht[:,m_:M_bht-1] = 1e6*H_bht[:,m_:M_bht-1]/C2/rho2

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

  # maximum dose as a function of time
  Dmax = np.zeros( (N+1,1), dtype=np.float64 )

  # print firing sequence
  if iverbose==1:
    print sequence
    
  steps = np.floor( np.linspace(0,N-2,10) )
  steps = steps.astype(int)

  # Integrate BHT:

  # output to screen
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
    T = np.squeeze(T)
    
    # compute slopes of Runge-Kutta scheme
    s1 = A.dot(T) 
    s1 += np.squeeze(sequence[n]*Q)

    #s2 = spsolve( B, A.dot(T+0.5*dt*s1) + np.squeeze(sequence[n+1]*Q) )
    if (ilu==1):
      s2 = solve2( A.dot(T+0.5*dt*s1) + np.squeeze(sequence[n+1]*Q) )
    else:
      s2 = spsolve( B, A.dot(T+0.5*dt*s1) + np.squeeze(sequence[n+1]*Q) )
    
    # update temperature vector
    T = T + 0.5*dt*(s1+s2)
    
    # get spatial peak temperature
    Tpeak[n+1] = np.max(T)
    
    # compute thermal dose
    D = np.squeeze(D)
    D += equivalent_time.equivalent_time( T, JM_bht )
    
    # get peak thermal dose
    Dmax[n+1]  = np.amax(D)

    # check whether spatial peak temp is maximal
    if (Tpeak[n+1]>tt):
      tt = np.float(Tpeak[n+1])
      Tmax_vec = T
      t_peak = t[n+1]
      
    # output to screen
    if n in steps:
      p2 = np.floor( 10.0*(n+1.0)/N )
      p1 = timing.timing( p1, p2, t_start, t[n], 1.0, n )

    # compute minimum time at which a thermal dose of delivered.
    if (np.min( np.abs( np.squeeze(Dmax[:,0]) - 240.0 ) ) < 0.001 ):
      min_itime = np.argmin( np.abs( Dmax[:,0] - 240.0 ) )
      min_time = t[min_itime]
      print '\n\tBeginning to lesion ... \t %f \n', min_time
      
  return residual
      