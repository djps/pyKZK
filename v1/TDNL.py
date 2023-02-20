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
  #U.astype(complex)
  #U.real.astype(complex)
    
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

  #del X,c,P,K,cfl,k,j
	
  return u,U,Ppos,Pneg,I_td 
