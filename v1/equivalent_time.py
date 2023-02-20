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
