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