import numpy as np
from scipy.sparse.linalg import spsolve, factorized, splu
from scipy.sparse import spdiags, eye

from scipy.sparse.linalg import LinearOperator


class ThermalGridClass():
    def __init__(self, Z, R, JJ, NN, r, z):
        self.Z = Z
        self.R = R
        self.JJ = np.int(JJ)
        self.NN = np.int(NN)
        self.r = r
        self.z = z


class GridClass():
    def __init__(self, Z, KK, R, JJ, NN, r, z):
        self.Z = Z
        self.KK = np.int(KK)
        self.R = R
        self.JJ = np.int(JJ)
        self.NN = np.int(NN)
        self.r = r
        self.z = z


def BuildBHTperipherals(Grid, Layer, Q, dt, verbose, rskip, zskip, Tvec, isFirst):
  """
  Builds the heating grid:
  """

  if (isFirst):
    r  = Grid.r[0:np.size(Grid.r) :rskip]
    z  = Grid.z[0:np.size(Grid.z) :zskip]
    #print( "isFirst Tvec max:", np.max(Tvec[:]) )
  else:
    r  = Grid.r
    z  = Grid.z

  KK = 0
  dr = r[1]
  JJ = np.size(r)
  dz = z[1] - z[0]
  NN = np.size(z)

  JN = JJ * NN

  Grid2 = GridClass(Grid.Z, KK, Grid.R, np.int(JJ), np.int(NN), r, z)

  Qvec = np.reshape(Q[0:np.size(r), 0:np.size(z)], JN)

  II = np.int( np.size(Layer) )

  # reporting
  if (verbose):
    if (isFirst):
      print('\n\tThermal Grid stepsize')
      print('\t\tdz = %3.2f [mm]' %(10*dz))
      print('\t\tdr = %3.2f [mm]\n' %(10*dr))

  # compute diffusivity d, the reciprocal of the perfusion time constant v, and the coefficient for the power density for all the tissue layers:
  for ii in np.arange(0,II):
    Layer[ii].d    = 1e4 * Layer[ii].kappa / Layer[ii].Cp / Layer[ii].rho	# units cm^2/s
    Layer[ii].v    = Layer[ii].w / Layer[ii].rho	# units 1/s
    Layer[ii].coef = 1e2 / Layer[ii].Cp / Layer[ii].rho # units K cm s^2/kg

  # build matrix operator's vector "bands"
  alpha_plus  = np.ones((JN,), dtype=np.float)/dz/dz
  alpha_minus = np.ones((JN,), dtype=np.float)/dz/dz
  bp = np.zeros((JJ,), dtype=np.float)
  bm = np.zeros((JJ,), dtype=np.float)
  bp[0] = 2.0 / dr / dr
  bp[1:JJ] = (1.0/dr + 0.5 / Grid2.r[1:JJ]) / dr
  bm[1:JJ] = (1.0/dr - 0.5 / Grid2.r[1:JJ]) / dr
  beta_plus  = np.zeros((JN,), dtype=np.float)
  beta_minus = np.zeros((JN,), dtype=np.float)

  for jn in np.arange(0, JN, dtype=np.int):
    if (np.mod(jn+1, NN) == 0):
      alpha_plus[jn] = 0
    if (np.mod(jn+1, NN) == 1):
      alpha_minus[jn] = 0
    qq = np.int( np.ceil(jn / NN) - 1)
    beta_plus[jn]  = bp[qq]
    beta_minus[jn] = bm[qq]

  gamma = np.squeeze( -2.0*(1.0/dr/dr + 1.0/dz/dz)*np.ones( (JN,1), dtype=np.float) )

  z_t = np.zeros((II,), dtype=np.float)
  ii = np.int(0)

  # build diffusivity and perfusion coefficient matrices (describe layers)
  for ii in np.arange(0, II, dtype=np.int):
    z_t[ii] = Layer[ii].z

  if (z_t[0] < z[0]):
    ii = np.int(1)
  else:
    ii = np.int(0)

  d = np.zeros( (NN,), dtype=np.float )
  v = np.zeros( (NN,), dtype=np.float )
  c = np.zeros( (NN,), dtype=np.float )

  # for each point
  for nn in np.arange(0, NN, dtype=np.int):
    # fill by layer
    if (np.min( np.abs( z[nn] - z_t ) ) < dz/2.0):
        ii = np.int(ii+0)
    else:
        ii = np.int(ii+0)
    # now fill properties
    d[nn] = Layer[ii].d
    v[nn] = Layer[ii].v
    c[nn] = Layer[ii].coef

  D = d
  V = v
  C = c
  print("C:", np.size(C))

  for _ in np.arange(0, JJ-1):
    D = np.hstack((D, d))
    V = np.hstack((V, v))
    C = np.hstack((C, c))
  print("C:", np.size(C))

  D = np.transpose(D)
  V = np.transpose(V)
  C = np.transpose(C)
  print("C:", np.size(C))

  D = spdiags(D, 0, JN, JN)
  V = spdiags(V, 0, JN, JN)

  dt = np.float(dt)

  I = eye(np.int(JN), np.int(JN), dtype=np.float)

  temp = np.array([np.squeeze(beta_plus), np.squeeze(alpha_plus), np.squeeze(gamma), np.squeeze(alpha_minus), np.squeeze(beta_minus)], dtype=np.float )
  A = spdiags(temp, [-NN, -1, 0, 1, NN], JN, JN)
  A = A.T

  # Crank-Nicolson operators
  temp = dt * (D.multiply(A) - V) / np.float(2.0)
  CN1 = I - temp
  CN2 = I + temp

  # rescale power density to obtain heating rate
  Hvec = C * Qvec

  Hvec = dt * Hvec
  Hvec = Hvec.astype(np.float, copy=False)

  return CN1, CN2, Hvec, Grid2


def BuildBHTperipheralsTemperature(Grid, TemperatureLayer, Q, dt, verbose, rskip, zskip, Tvec, isFirst=False):
  """
  Builds the heating grid:
  """

  if (isFirst):
    r  = Grid.r[0 : np.size(Grid.r) : rskip]
    z  = Grid.z[0 : np.size(Grid.z) : zskip]
    print( "isFirst Tvec max:", np.max(Tvec[:]) )
  else:
    r  = Grid.r
    z  = Grid.z

  KK = 0
  dr = r[1]
  JJ = np.size(r)
  dz = z[1] - z[0]
  NN = np.size(z)

  JN = JJ * NN

  Grid2 = GridClass(Grid.Z, KK, Grid.R, np.int(JJ), np.int(NN), r, z)

  Qvec = np.reshape(Q[0:np.size(r), 0:np.size(z)], JN)

  II = np.int( np.size(TemperatureLayer) )
  print("np.int( np.size(TemperatureLayer) ):", np.int( np.size(TemperatureLayer) ) )
  j = 0
  for i in np.arange(0,II):
    j = j + np.size(TemperatureLayer[i].rho)
    print("data:", np.size(TemperatureLayer[i].rho), "j:", j, "j/JJ:", j/JJ, "j/NN:", j/NN)

  # reporting
  if (verbose):
    if (isFirst):
      print('\tThermal Grid stepsize')
      print('\t\tdz = %3.2f [mm]' %(10*dz))
      print('\t\tdr = %3.2f [mm]' %(10*dr))
      print('\t\tdt = %3.2f [s]\n' %dt)

  # compute diffusivity d, the reciprocal of the perfusion time constant v, and the coefficient for the power density for all the tissue layers:
  for ii in np.arange(0,II):
    TemperatureLayer[ii].d    = 1e4 * TemperatureLayer[ii].kappa / TemperatureLayer[ii].Cp / TemperatureLayer[ii].rho	# units cm^2/s
    TemperatureLayer[ii].v    = TemperatureLayer[ii].w / TemperatureLayer[ii].rho	# units 1/s
    TemperatureLayer[ii].coef = 1e2 / TemperatureLayer[ii].Cp / TemperatureLayer[ii].rho # units K cm s^2/kg

  # build matrix operator's vector "bands"
  alpha_plus  = np.ones((JN,), dtype=np.float)/dz/dz
  alpha_minus = np.ones((JN,), dtype=np.float)/dz/dz
  bp = np.zeros((JJ,), dtype=np.float)
  bm = np.zeros((JJ,), dtype=np.float)
  bp[0] = 2.0 / dr / dr
  bp[1:JJ+1] = (1.0/dr + 0.5 / Grid2.r[1:JJ+1]) / dr
  bm[1:JJ+1] = (1.0/dr - 0.5 / Grid2.r[1:JJ+1]) / dr
  beta_plus  = np.zeros((JN,), dtype=np.float)
  beta_minus = np.zeros((JN,), dtype=np.float)

  for jn in np.arange(0, JN, dtype=np.int):
    if (np.mod(jn+1, NN) == 0):
      alpha_plus[jn] = 0
    if (np.mod(jn+1, NN) == 1):
      alpha_minus[jn] = 0
    qq = np.int( np.ceil(jn / NN) - 1)
    beta_plus[jn]  = bp[qq]
    beta_minus[jn] = bm[qq]

  gamma = np.squeeze( -2.0*(1.0/dr/dr + 1.0/dz/dz)*np.ones( (JN,1), dtype=np.float) )

  z_t = np.zeros((II,), dtype=np.float)
  ii = np.int(0)

  # build diffusivity and perfusion coefficient matrices (describe layers)
  for ii in np.arange(0, II, dtype=np.int):
    z_t[ii] = TemperatureLayer[ii].zlayer[1]

  D = TemperatureLayer[0].d
  V = TemperatureLayer[0].v
  C = TemperatureLayer[0].coef
  for j in np.arange(1, II, dtype=np.int):
    D = np.hstack((D, TemperatureLayer[j].d) )
    V = np.hstack((V, TemperatureLayer[j].v) )
    C = np.hstack((C, TemperatureLayer[j].coef) )
  D = np.transpose(D)
  V = np.transpose(V)
  C = np.transpose(C)

  print( "sizes. D:", np.shape(D), "V:", np.shape(V), "C:", np.shape(C), "JJ:", JJ, "NN:", NN, "JN:", JN, "Qvec:", np.shape(Qvec) )

  # # set counter ii for starting layer
  # if (z_t[0] < z[0]):
  #   ii = np.int(1)
  # else:
  #   ii = np.int(0)
  # offset = 0
  # j = np.int(0)

  # # for each point
  # for nn in np.arange(0, NN, dtype=np.int):
  #   # fill by layer
  #   if (np.min( np.abs( z[nn] - z_t ) ) < dz/2.0):
  #     ii = np.int(ii)
  #     print("new layer")
  #     if (ii > 0):
  #       offset = offset + np.size( TemperatureLayer[ii-1].d )
  #     print("offset is:", offset)
  #     j = np.int(0)
  #   j = np.int(j + 1)
  #   d[nn] = TemperatureLayer[ii].d[ offset + j]
  #   v[nn] = TemperatureLayer[ii].v[ offset + j]
  #   c[nn] = TemperatureLayer[ii].coef[ offset + j]
  # D = d
  # V = v
  # C = c
  # for _ in np.arange(0, JJ-1):
  #   D = np.hstack((D, d))
  #   V = np.hstack((V, v))
  #   C = np.hstack((C, c))
  # D = np.transpose(D)
  # V = np.transpose(V)
  # C = np.transpose(C)
  # print("---", np.size(D), "---")

  D = spdiags(D, 0, JN, JN)
  V = spdiags(V, 0, JN, JN)

  dt = np.float(dt)

  I = eye(np.int(JN), np.int(JN), dtype=np.float)

  temp = np.array([np.squeeze(beta_plus), np.squeeze(alpha_plus), np.squeeze(gamma), np.squeeze(alpha_minus), np.squeeze(beta_minus)], dtype=np.float )
  A = spdiags(temp, [-NN, -1, 0, 1, NN], JN, JN)
  A = A.T

  # Crank-Nicolson operators
  temp = dt * (D.multiply(A) - V) / np.float(2.0)
  CN1 = I - temp
  CN2 = I + temp

  # rescale power density to obtain heating rate
  print("diagnostic | C:", np.shape(C), ", Qvec:", np.shape(Qvec), ", diff(C,Q):", np.size(C)-np.size(Qvec), ", JJ;" ,JJ, ", NN:", NN, ", JN:", JN)
  Hvec = C * Qvec

  Hvec = dt * Hvec
  Hvec = Hvec.astype(np.float, copy=False)

  return CN1, CN2, Hvec, Grid2
