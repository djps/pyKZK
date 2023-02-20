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
