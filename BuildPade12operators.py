from scipy.sparse import eye
import numpy as np

def BuildPade12operators(A,kk,dz,k,JJ):
  """
  Builds pade operators at (1,2)-order
  """
  
  I = eye(JJ, dtype=np.complex, format='dia')  
  kkk = k * kk
  
  A = A / kkk / kkk
  
  s = 1j * kkk * dz
  
  muplus  = (3 - 2*s*s + 1j * np.sqrt((((2*s+6)*s-6)*s-18)*s-9))/12/(1+s)
  muminus = (3 - 2*s*s - 1j * np.sqrt((((2*s+6)*s-6)*s-18)*s-9))/12/(1+s)
  epsilon = ((s+3)*s+3)/6/(1+s)
  
  P1 = I + muplus*A
  P2 = I + muminus*A
  P3 = I + epsilon*A
  
  return P1, P2, P3
