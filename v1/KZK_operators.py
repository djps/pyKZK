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
  
  #CN1 = splu(CN1)
  #CN2 = splu(CN2)
  
  #if (k==1): 
    #np.savetxt('w.out', w.view(float).reshape(-1, 2) )
    #np.savetxt('ReA.out', ReA.todense() )
    #np.savetxt('ImA.out', ImA.todense() )
    #np.savetxt('alpha.out', alpha.todense() )
    #np.savetxt('beta.out', beta.todense() )
    
  #if k==1:
    #filename = os.path.expanduser('~/Desktop/tempmat.mat')
    #data = loadmat(filename)
    #IRK11mat = data['tempmat1']
    #IRK12mat = data['tempmat2']
    
    #ReAmat = data['tempmat5']
    #ImAmat = data['tempmat6']
    
    #alphamat = data['tempmat7']
    #betamat = data['tempmat8']
    
    #alphagamma2mat = data['tempmat11']
    #betagamma2mat = data['tempmat12']
    
    #print type(test1), type(IRK11[1]), type(test2), type(IRK12[1]), '\n'
    #print '\n', np.shape(test1), np.shape(test2), np.shape(IRK11[1]), np.shape(IRK12[1]), '\n'
    #print gamma, np.real(gamma), np.imag(gamma)
    
    #print "1'=>", IRK11mat[0,0], IRK1[0,0], "<=", '\n'
    #print "2'=>", IRK12mat[0,0], IRK2[0,0], "<=", '\n'
    #print "3'=>", ReAmat[0,0], ReA[0,0], "<=", '\n'
    #print "4'=>", ImAmat[0,0], ImA[0,0], "<=", '\n'
    #print "5'=>", alphamat[0,0], alpha[0,0], "<=", '\n'
    #print "6'=>", betamat[0,0], beta[0,0], "<=", '\n'
    
    #print "7'=>", alphagamma2mat[0,0], alpha[0,0], "<=", '\n'
    #print "8'=>", betagamma2mat[0,0], beta[0,0], "<=", '\n'
     
    #print "=>", test2[0,0],  1.0*IRK12[1][0,0], "<=", '\n'
    #print "\ttest1", test1[0:15,0:15], '\n'
    #print "\tIRK11",  1.0*IRK11[1][0:15,0:15], '\n'
    #print "\ttest2", test2[0:15,0:15], '\n'
    #print "\tIRK12",  1.0*IRK12[1][0:15,0:15], '\n'
    #print np.shape(test1.data),np.shape(IRK11[1].data), np.shape(test2.data),np.shape(IRK12[1].data), '\n'
    #print np.shape(test1.indices), np.shape(IRK11[1].indices), np.shape(test2.indices), np.shape(IRK12[1].indices), '\n'
    #print np.shape(test1.indptr), np.shape(IRK11[1].indptr), np.shape(test2.indptr), np.shape(IRK12[1].indptr), '\n'
    #print np.max( np.max( np.abs( test1 -  1.0*IRK11[1] ) ) ), '\n'
    #print np.max( np.max( np.abs( test2 -  1.0*IRK12[1] ) ) ), '\n'
    #print "***", test1[0,J], test1[0,J+1], 1.0*IRK11[1][0,J], 1.0*IRK11[1][0,J+1], "***", '\n'
    #print "***", test2[0,J], test2[0,J+1], 1.0*IRK12[1][0,J], 1.0*IRK12[1][0,J+1], "***", '\n'
  
  del I,ReA,ImA,alpha,beta,M,A,supdiag,diagonal,subdiag,w,j,rr
  
  return IRK1,IRK2,CN1,CN2