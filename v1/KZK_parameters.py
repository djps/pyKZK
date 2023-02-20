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
  ztissue = 2.4
  
  # material transition distance [cm]
  z_ = d - ztissue 

  # number of harmonics included in simulation
  K = 36
  
  # dictionary containing dBm levels and powers in Watts
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
  
  # dictionary containing dBm levels and appropriate numbers of harmonics to compute.
  dbm_harmonics = { \
  '-22': 12, \
  '-20': 16, \
  '-19': 20, \
  '-18': 26, \
  '-16': 30, \
  '-15': 34, \
  '-14': 36, \
  '-13': 38, \
  '-12': 40, \
  '-11': 48, \
  '-10': 48, \
  '-9' : 48, \
  '-8' : 48, \
  '-7' : 48, \
  '-6' : 48, \
  '-5' : 48}
  
  # token value for minimum difference between drive and power output at a given drive setting 
  mindiff = 10.0
  
  # infer from drive settings the appropriate number of harmonics to compute.
  #for value in dbm.values(): 
  for setting0, value in dbm.iteritems():
    diff = np.abs( value - drive )
    if (diff < mindiff):
	diff = mindiff
        setting = setting0
        
  # number of harmonics included in simulation
  K = np.int( dbm_harmonics[setting] )

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
  
  #print gamma2
  
  return p0,c1,c2,rho1,rho2,N1,N2,G1,G2,gamma1,gamma2,alpha1,alpha2,a,b,d,f,R,Z,z_,K
