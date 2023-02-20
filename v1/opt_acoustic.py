#--------------------------------------------------------------------------------------------------
#
def KZK_operators(r,R,G,dz,dr,J,k,gamma):
#
#-------------------------------------------------------------------------------------------------- 

  import numpy as np
  import os
  from scipy.io import loadmat, savemat
  from scipy.sparse import spdiags, eye, bmat
  from scipy.sparse.linalg import splu, factorized
  from math import pi
  
  w = np.zeros( (J,1) )
  for j in np.arange(0,J):
    if ( r[j] > R ): 
      w[j] = 16.0*( r[j] - R )**2
 
  w = np.squeeze( np.exp( 1j*pi*w/2.0 ) )
  
  k = k+1
  
  supdiag  = np.squeeze(np.zeros((J,1)))
  diagonal = np.squeeze(np.zeros((J,1)))
  subdiag  = np.squeeze(np.zeros((J,1)))  

  rr = np.copy(r)
  rr[0] = 1.0
  supdiag  = np.conjugate( ( w/dr + 0.5/rr )/dr )
  diagonal = np.conjugate( -2.0*w/dr/dr ) 
  subdiag  = np.conjugate( ( w/dr - 0.5/rr )/dr )
    
  temp = np.array( [supdiag, diagonal, subdiag] )
  
  A = spdiags(temp, np.array([-1,0,1]), J, J)
    
  A = A.transpose()

  A = A.conjugate()

  A[0,1] = -A[0,0]

  ReA = A.real
  ReA = 0.25*ReA/G/k
  
  ImA = A.imag
  ImA = 0.25*ImA/G/k

  alpha = np.real(gamma)*eye(J, J, dtype=np.float64)
  beta  = np.imag(gamma)*eye(J, J, dtype=np.float64)
  
  alpha = alpha.tocsc()
  beta  = beta.tocsc()
  
  M = bmat( [ [ImA-alpha, -ReA-beta], [ReA+beta, ImA-alpha] ] )
  
  I = eye(2*J, 2*J, dtype=np.float64)

  IRK1 = I - dz*( 1.0 - 1.0/np.sqrt(2.0) )*M
  IRK2 = M
  
  IRK1 = IRK1.tocsc()
  IRK2 = IRK2.tocsc()
  
  #IRK1 = splu(IRK1)
  #IRK2 = splu(IRK2)
  
  IRK1 = factorized( IRK1 )

  CN1 = I - 0.5*dz*M
  CN2 = I + 0.5*dz*M
  
  CN1 = CN1.tocsc()
  CN2 = CN2.tocsc()
    
  del I,ReA,ImA,alpha,beta,M,A,supdiag,diagonal,subdiag,w,j,rr
  
  return IRK1,IRK2,CN1,CN2
  
#--------------------------------------------------------------------------------------------------
#
def TDNL(u,U,X,K,J,c,cutoff,I_td):
#
#--------------------------------------------------------------------------------------------------

  import numpy as np
  import warnings
  from numpy.fft import fft, ifft
    
  warnings.simplefilter("ignore", np.ComplexWarning)
    
  if (K==1):
    print "linear!"
  else:
    for j in np.linspace(J-1, 0, num=J):
      I_td[j] = 0
      if ( (np.sqrt(u[j,0]**2+u[J+j,0]**2) > cutoff) or (j==1) ):
	U[ 0, 0 ] = 0.0
	U[ 0, 1 : K+1 ] = np.real(u[j,:]) - 1j*np.real(u[j+J,:])
	U[ 0, 2*K-1 : K : -1 ] = np.real(u[j,0:K-1]) + 1j*np.real(u[j+J,0:K-1])
	U = K*np.real( ifft(U) )
	I_td[j] = np.trapz( np.squeeze(U**2) )
	cfl = 0.8
	P = np.ceil( c*np.max(np.squeeze(U))/cfl )
	for p in np.arange(0,P):
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
	  U = X
	I_td[j] = I_td[j] - np.trapz( np.squeeze( np.real(X) )**2 ) 
	X = fft(X)/K
	u[j,:]   = np.real( X[0,1:K+1] )
	u[j+J,:] =-np.imag( X[0,1:K+1] )
	
  return u,U,I_td 

#--------------------------------------------------------------------------------------------------
#
def func(param, drive, efficiency, filename):
#
#--------------------------------------------------------------------------------------------------
    
  import numpy as np
  
  import matplotlib.pyplot as plt
  import matplotlib.mlab as mlab
  
  import time, os, warnings
  
  from math import pi
  from scipy.io import loadmat, savemat
  
  from scipy import transpose
  
  from scipy.sparse import spdiags, eye
    
  from scipy.sparse.linalg import spsolve, factorized, splu
  
  from scipy.linalg import solve, lu
  
  from opt_acoustic import TDNL, KZK_operators

  # drive power [Watts]
  drive = 3.0

  # efficiency of transducer [dimensionless]
  efficiency = 0.75

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
  
  # small-signal sound speed [m/s]
  c2      = 1570.0 
  
  # mass density [kg/m^3]
  rho2    = 1070.0 
  
  # attenuation at 1MHz [dB/m]
  alpha02 = param[0]
  
  # power of attenuation vs frequency curve
  eta2    = param[1]
  
  # nonlinearity parameter [dimensionless]
  beta2   = param[2]
  
  # absorbtion at 1MHz [dB/m]
  theta02 = 33  
  
  # power of absorbtion vs frequency curve
  phi2    = 1.1 
  
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
  
  # max radius [cm]
  R = a 
  
  # max axial distance [cm]
  Z = 1.5*d 
  
  # propation distance through tissue [cm]
  ztissue = 2.24
  
  # material transition distance [cm]
  z_ = d - ztissue 

  # number of harmonics included in simulation
  K = 36
  
  # heat capacity [J/kg/K]
  Cv1 = 4180.0 
  
  # thermal conductivity [W/m/K]
  kappa1 = 0.6  
  
  # perfusion rate [kg/m^3/s]
  w1 = 0.0  
  
  # heat capacity [J/kg/K]
  Cv2 = param[3]
  
  # thermal conductivity [W/m/K]
  kappa2 = param[4]
  
  # perfusion rate [kg/m^3/s] : Med. Phys. (2000) 27(5) p.1131-1140, use 0.5-20
  w2 = 0.0   

  # ambient temperature [degrees C]
  T0 = 37.0
  
  # reduction factor in r-direction
  r_skip = 2
  
  # reduction factor in z-direction
  z_skip = 4
  
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
  
  #--------------------------------------------------------------------------------------------------
  # formatting of experimental data
  
  # get data to fit against
  if os.path.isfile(filename):
    data = np.genfromtxt(filename,skip_header=3,usecols=(0,2))
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
  
  #------------------------------------------------------------------------------------------------
  
  # set size of arrays which store complex values for the K harmonics
  K2 = 2*K

  # check F-number satisfies paraxial criteria
  F = 0.5*d/a
  if (F<1.37):
    print ' *** Warning -- f/%1.2f exceeds the conditions under which KZK is derived (> f/1.37) *** \n' %(F)
    
  #------------------------------------------------------------------------------------------------
    
  # est. points per wavelength in axial direction
  ppw_ax = 40
  
  # number of meshpoints in axial direction 
  M = np.around(ppw_ax*Z*max(G1,G2))

  # axial stepsize
  dz = Z/(M-1.0)

  # axial node vector
  z = np.linspace(0, Z, num=np.int(M))    

  # est. points per wavelength in radial direction
  ppw_rad = 50
  
  # number of meshpoints in radial direction   
  J = np.around(ppw_rad*R*max(G1,G2)/pi) 

  # [0,R] is physical, the extra 0.25 is for PML
  R_ext = R+0.25 
  dr = R_ext/(J-1.0)

  # radial node vector
  r = np.linspace(0.0, R_ext, num=np.int(J))
    
  # number of nodes in [0,R]
  J_ = np.ceil( J*R/R_ext )
  
  #------------------------------------------------------------------------------------------------
  
  # dependent variables u, v
  u = np.zeros( (2*J,K) )
  v = np.zeros( (2*J,K) )
  
  limit = 1.0/np.sqrt( 1.0 - (a/d)**2 )
  
  #------------------------------------------------------------------------------------------------

  # j runs from 0 to J-1, so can be used to populate an array with J values.
  for j in np.arange(0,J):
    if ( (np.abs(r[j]) >= b*limit/a) and (np.abs( r[j] ) <= limit) ):
      arg      = G1*r[j]*r[j]
      v[j,0]   = np.cos(arg) 
      v[J+j,0] =-np.sin(arg) 
      
  #------------------------------------------------------------------------------------------------
  
  # format v
  v[0:J,0]   = v[0:J,0]*np.sqrt( 1.0-(r/d)**2 )  
  v[J:2*J,0] = v[J:2*J,0]*np.sqrt( 1.0 - (r/d)**2 ) 
  
  op = [KZK_operators(r,R,G1,dz,dr,J,k,gamma1[k]) for k in np.arange(0,K)] 
  IRK11, IRK12, CN11, CN12 = zip(*op)
  
  op = [KZK_operators(r,R,G1,dz,dr,J,k,gamma2[k]) for k in np.arange(0,K)] 
  IRK21, IRK22, CN21, CN22 = zip(*op)
       
  # IRK slope vectors
  k1 = np.zeros( (2*J,1) )
  k2 = np.zeros( (2*J,1) )

  # IRK coefficients
  b1 = 1.0/np.sqrt(2.0)
  b2 = 1.0 - b1
  
  # nonlinear term integration parameters
  mu1 = N1*K*dz/pi         
  mu2 = N2*K*dz/pi
  
  # cutoffs for performing nonlinear integration
  cutoff1 = gamma1[0]/10.0/N1
  cutoff2 = gamma2[0]/10.0/N2
  
  # data vectors
  X = np.zeros((1,K2)) #, dtype=complex)
  Y = np.zeros((1,K2), dtype=complex)
    
  # Heating rate matrices
  H  = np.zeros((J_,M+1))
  H2 = np.zeros((J_,M+1))
  H[:,0] = np.real( gamma1[0] )*( v[0:J_,0]**2 + v[J:J+J_,0]**2 )

  # axial intensity vector
  Ix = np.zeros((M+1,1))
  Ix[0] = v[0,0]**2 + v[J,0]**2
  
  I_td = np.zeros((J,1))
  
  dt = 1.0/f/(K2-1.0)
  
  # first (up to) 5 harmonic pressure amplitudes in axial direction
  if (K<5): 
    kk = K 
  else:
    kk = 5
  
  # index of last meshpoint in material 1? ceil rounds up
  m_t = np.ceil(z_/dz/d)
  if(m_t>M): 
    m_t = M 

  # index of meshpoint nearest focus   
  m_f = np.around(M/Z)
    
  #--------------------------------------------------------------------------------------------------

  # spatial front: from 0 to m_t-1
  for m in np.arange(0,m_t):
    # compute updated v
    v,X,I_td = TDNL(v,X,Y,K,J,mu1,cutoff1,I_td)
    # update time-domain intensity
    I_td = f*dt*I_td/dz
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
    # update heating rates
    for j in np.arange(0,J_):
      H[j,m+1]  = np.sum( np.real( np.transpose(gamma1[:]) )*( u[j,:]**2 + u[J+j,:]**2 ) )
      H2[j,m+1] = I_td[j]
    # gain check
    Ix[m+1] = sum( v[0,:]**2 + v[J,:]**2 )
    if ( Ix[m+1] > 2*G1*G1 ):
      print '\tStopped - computation became unstable at z = %2.1f cm.\n' %(d*z[m])
      r = a*r[0:J_]
      z = d*z
      exit(0)
  
  # impediance mismatch loss
  v = 2.0*rho2*c2*v/(rho1*c1+rho2*c2)
  
  for m in np.arange(m_t,M):
    v,X,I_td = TDNL(v,X,Y,K,J,mu2,cutoff2,I_td)
    I_td = f*dt*I_td/dz   
    if (z[m]<0.3):
      if (m==m_t):
	for k in np.arange(0,K):
	  #k1 = spsolve(IRK11[k], IRK12[k]*v[:,k] )
	  #k2 = spsolve(IRK21[k], IRK22[k]*(v[:,k] + dz*b1*k1) ) 
	  k1 = IRK11[k]( IRK12[k]*v[:,k] )
	  k2 = IRK21[k]( IRK22[k]*(v[:,k] + dz*b1*k1) )
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
    
    for j in np.arange(0,J_):
      H[j,m+1]  = sum( np.real( np.transpose(gamma2[:]) )*( u[j,:]**2 + u[J+j,:]**2) )
      H2[j,m+1] = I_td[j]

    Ix[m+1] = sum( v[0,:]**2 + v[J,:]**2 )
    if ( Ix[m+1] > 2*G2*G2 ):
      print '\tStopped - computation became unstable at z = %2.1f cm.\n' %(d*z[m])
      r = a*r[0:J_]
      z = d*z
      exit(0)
    
  # dimensionalize H
  H[:,0:m_t+1]  = 1e-4*p0*p0*H[:,0:m_t+1]/rho1/c1/d
  H[:,m_t+1:M]  = 1e-4*p0*p0*H[:,m_t+1:M]/rho2/c2/d

  # dimensionalize H2
  H2[:,0:m_t+1] = 1e-4*0.5*p0*p0*H2[:,0:m_t+1]/rho1/c1/d
  H2[:,m_t+1:M] = 1e-4*0.5*p0*p0*H2[:,m_t+1:M]/rho2/c2/d

  # combine H
  H = np.real(H + H2)

  # rescale r and chop so that PML region is excluded in plots
  r = a*r[0:J_]

  # rescale z 
  z = d*z

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
    
  # nondimensional diffusivity of material 1 [dimensionless]
  D1 = 1e4*kappa1/Cv1/rho1

  # nondimensional diffusivity of material 2 [dimensionless]
  D2 = 1e4*kappa2/Cv2/rho2

  # nondimensional perfusion/density of material 1 [dimensionless]
  P1 = w1/rho1

  # nondimensional perfusion/density of material 2 [dimensionless]
  P2 = w2/rho2

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
  for jm in np.arange(0,JM_bht):
    if (np.mod(jm+1,M_bht)==0): 
      alpha_plus[jm] = 0.0   
    if (np.mod(jm+1,M_bht)==1): 
      alpha_minus[jm] = 0.0
    jn = np.ceil( (jm+0)/M_bht )
    beta_plus[jm]  = bp[jn]
    beta_minus[jm] = bm[jn]
  gamma_0 = np.squeeze( -2.0*(1.0/dr/dr + 1.0/dz/dz)*np.ones( (JM_bht,1), dtype=np.float64) )

  # build diffusivity coefficient matrix
  k1minor = D1*np.ones( (m_,1), dtype=np.float64)
  k2minor = D2*np.ones( (M_bht-m_,1), dtype=np.float64)
  k = np.vstack((k1minor,k2minor))
  bigk = k
  for j in np.arange(0,J_bht-1):
    bigk = np.vstack((bigk,k))
  Kmat = spdiags(np.squeeze(bigk), 0, JM_bht, JM_bht );

  # build perfusion coefficeint matrix:
  p1 = P1*np.ones( (m_,1), dtype=np.float64 )
  p2 = P2*np.ones( (M_bht-m_,1), dtype=np.float64 )
  pp  = np.vstack((p1,p2))
  bigp = pp
  for j in np.arange(0,J-1):
    bigp = np.vstack((bigp,pp))
  Pmat = spdiags(np.squeeze(bigp), 0, JM_bht, JM_bht)
  
  # create matrix A 
  rows = np.array([np.squeeze(beta_plus),np.squeeze(alpha_plus),np.squeeze(gamma_0),np.squeeze(alpha_minus),np.squeeze(beta_minus)], dtype=np.float64 )
  positions = np.array([-M_bht, -1, 0, 1, M_bht], dtype=np.float64 )
  A = spdiags( rows, positions, JM_bht, JM_bht )
  A = A.T
  A = A*Kmat - Pmat
  
  # create matrix B
  B = eye(JM_bht,JM_bht) - 0.5*dt*A
  
  del alpha_plus,alpha_minus,bp,bm,beta_plus,beta_minus,gamma_0,k1minor,k2minor,Kmat,k,bigk,p1,p2,pp,bigp,Pmat,rows,positions
   
  # accelerate algorithm by prefactoring
  solve1 = factorized(A)
  solve2 = factorized(B)

  # rescale H_bht to degrees/second
  H_bht[:,0:m_-1]     = 1e6*H_bht[:,0:m_-1]/Cv1/rho1
  H_bht[:,m_:M_bht-1] = 1e6*H_bht[:,m_:M_bht-1]/Cv2/rho2

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

  # integration of BHT equation
  for n in np.arange(0,N-1):
    n = int(n)
    sequence = np.squeeze(sequence)
    T        = np.squeeze(T)
    s1 = A.dot(T) 
    s1 += np.squeeze(sequence[n]*Q)
    #if (ilu==1):
    s2 = solve2( A.dot(T+0.5*dt*s1) + np.squeeze(sequence[n+1]*Q) )
    #else:
      #s2 = spsolve( B, A.dot(T+0.5*dt*s1) + np.squeeze(sequence[n+1]*Q) )
    T = T + 0.5*dt*(s1+s2)
    Tpeak[n] = np.max(T)
  
  err = ydata - np.squeeze(Tpeak)  
  
  print np.abs( np.sum( err ) )
  
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
  plot1, = ax1.plot( tdata, ydata, linewidth=2, linestyle='-', color=ICRcolors['ICRred'], label='Experimental' )
  ax1.plot( tdata, ydata, marker='o', color=ICRcolors['ICRred'] )
  plot2, = ax1.plot( tdata, np.squeeze(Tpeak[0:N]), linewidth=2, linestyle='-', color=ICRcolors['ICRblue'], label='Least Squares' )
  ax1.plot( tdata, np.squeeze(Tpeak[0:N]), marker='o', color=ICRcolors['ICRblue'] )
  
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
  
  # set axis for plot
  ax1.legend( [plot1, plot2], [r'Experimental', r'Least Squares'] )
  
  # apply grid to figure
  plt.grid(True)
  
  # render figure
  plt.show()   
      
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

alpha02 = 27.1

eta2 = 1.223

beta2 = 4.3525

# heat capacity [J/kg/K]
Cv2 = 3750.0

# thermal conductivity [W/m/K]
kappa2 = 0.58   

# input parameters
parameters = [ alpha02, eta2, beta2, Cv2, kappa2 ]

# tolerances
tol = np.array( [1e-06, 1e-06] )

# drive voltage
drive = 3.0

# total efficiency
efficiency = 0.74

# data set
filename='20dBm_1_0_0_0_20140516_4s_1c.txt'

# least squares minimization
pfin, cov_x, infodict, mesg, ierr = leastsq(func, parameters, args=(drive, efficiency, filename), Dfun=None, full_output=1, col_deriv=0, ftol=tol[0], xtol=tol[1] )
      