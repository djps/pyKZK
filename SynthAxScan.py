import numpy as np

def SynthAxScan(r, p, b, JJ_, nt, nharmonics = 5, verbose=False):
  """
    returns p_r and p_c, peak averaged rarefactional and compressional pressure and amplitude of first (up to) 5 averaged pressure harmonics

    r = radial node vector (cm)
    p = pressure matrix (radial x harmonic spectrum)
    b = hydrophone element radius (cm)
  """

  # JJ = number of radial nodes; KK = number of harmonics
  _, KK = np.shape(p)
  
  debug = False
  
  if (debug): print("shape p: ", np.shape(p) )

  # mesh spacing near axis
  dr_min = r[1]

  # vector of spatially averaged pressure values
  p_h = np.zeros((KK,), dtype=np.complex )

  # number of points over which to spatially average
  nmax = 4
  NN = np.max([nmax, np.ceil(10*b/dr_min)])

  dr = b / (NN - 1)

  x  = np.linspace(0.0, b, np.int(NN) )
  
  q  = np.zeros((np.int(NN),np.int(KK)), dtype=np.complex )
  
  U  = np.zeros((nt,), dtype=np.complex)

  if (debug): print( np.shape(x), "\t", np.shape(p[:,0])  )

  # for each harmonic, interpolate over radius of hydrophone then integrate
  for kk in np.arange(0, KK):
    q[:,kk] = np.interp(x, r, p[:,kk]) # is this both real and complex???
    p_h[kk] = dr * np.trapz( q[:,kk] * np.transpose(x) )

  if (np.abs(b) > 0.0):
    if (debug): print("b:", b)
    p_h = 2.0 * p_h / b / b
    p5  = p_h[0:np.min([nharmonics,KK])]
  else:
    print("this has been called incorrectly")
    p5  = p_h[0:np.min([nharmonics,KK])]
  
  # determine peak compressional p_c and rarefactional p_r pressure
  if (KK == 1):
    # linear case - do nothing
    p_c = np.abs(p_h[0])
    p_r = -p_c
  else:
    # nonlinear case - transform to time domain
    start = 0
    step = 1
    stop = KK
    if (debug): print( start, step, stop, np.shape(U[start:stop:step]), np.shape(np.conjugate(p_h)) ) 
    U[start:stop:step] = np.conjugate(p_h)
    start = 2*KK
    step = -1
    stop = KK-1
    if (debug): print( start, step, stop, np.shape(U[start:stop:step]), np.shape(p_h[0:KK]) ) 
    U[start:stop:step] = p_h[0:KK]
    U[0] = 0.0
    # transform to time domain:
    U = KK * np.real( np.fft.ifft(U) )
    p_r = np.min(U)
    p_c = np.max(U)

  return p_r, p_c, p5
