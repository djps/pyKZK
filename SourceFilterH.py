from scipy.special import jv as jv

import numpy as np

def SourceFilterH(x, A, k, verbose=False):

  # get some vital parameters
  JJ = np.size(x)
  R  = x[-1]

  # number of Newton iterations
  # tolerance of Newton process
  nmax = 5
  eps  = 10E-6

  # find JJ zeros of BesselJ(0,r)
  JJplus = JJ + 1
  c = np.zeros((JJplus,))

  # iterator jj will run from 0:JJ
  for jj in np.arange(0, JJplus, dtype=np.int):
      # first guess based on asymptotic approximation
      y = np.pi * (4.0*np.double(jj) - 1.0) / 4.0
      # Newton iteration
      for _ in np.arange(0, nmax):
        #c[jj] = y + np.besselj(0,y) / np.besselj(1,y)
        c[jj] = y + jv(0,y) / jv(1,y)
        # check for convergence
        if ( np.abs(c[jj] - y) < eps ):
          # converged, so use c[jj]
          break
        else:
          # replace y with c[jj] for next iteration
          y = c[jj]

  ## Maximum spatial frequency
  #V = c[-1] / (2.0*np.pi*R)

  ## vector of radial nodes (nonuniform)
  #r = np.transpose(c[-2]) * R / c[-1]

  # vector of spatial frequencies
  v = np.transpose(c[0:-1]) / (2.0*np.pi*R)

  [Jn, Jm] = np.meshgrid(c[0:-1], c[0:-1] )

  Jscale = Jn * Jm / c[-1]
  C = jv(0,Jscale) * (2.0 / c[-1]) / ( np.abs(jv(1,Jn)) * np.abs(jv(1,Jm)) )

  #temp = np.besselj(1, c[0:JJ])
  temp = jv(1, c[0:-1])
  temp = np.abs( temp )
  temp = temp / R
  m1 = np.transpose( temp  )

  ## used?
  #m2 = m1 * R / V

  # perform transforms and filtering
  q = 40
  s = 1.15
  F = (1.0 - np.tanh(q*(v/k - s/2.0/np.pi))) / 2.0

  if (verbose):
        print("SourceFilterH\n")
        print( "\tJJ : ", JJ,  "\tJn shape: ", np.shape(Jn),  "\tC shape: ", np.shape(C), '\tA shape: ', np.shape(A), '\tc shape: ', np.shape(c), '\tc[0:-2] shape: ', np.shape(c[0:-2]), '\tc[:-1] shape: ', np.shape(c[:JJ]), '\tc[1:JJplus] shape: ', np.shape(c[1:JJplus])  )

  del Jn
  del Jm

  # Apply Hankel transform
  if (verbose):
        print("shape m1 :", np.shape(m1) )
        print("shape A/m1 :", np.shape(A/m1) )
  Ahat = C @ (A / m1)
  if (verbose):
        print("size Ahat: ", np.shape(Ahat))

  # apply filter
  Ahatf = F * Ahat
  if (verbose):
        print("size F: ", np.shape(F))
        print("size Ahatf: ", np.shape(Ahatf))

  # apply inverse Hankel transform
  Af = (C @ Ahatf) * m1
  if (verbose):
        print("size Af: ", np.shape(Af))

  # return correct value
  return Af
