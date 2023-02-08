from scipy.sparse import eye
import numpy as np

def BuildPade11operators(A, kk, dz, k, JJ):
  """
  Builds pade operators at (1,1)-order
  """
  I   = eye(JJ, dtype=np.complex, format='dia')
  
  kkk = k * kk
  A   = A / kkk / kkk
  s   = 1j * kkk * dz
  P1  = I + 0.25 * (1.0 - s) * A
  P2  = I + 0.25 * (1.0 + s) * A
  
  return P1, P2
