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