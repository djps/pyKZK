
import BuildBHTperipherals, WAKZK_planar
import termcolor
import numpy as np
import os, time, datetime, warnings

from math import pi, log

from scipy.sparse import spdiags, eye

from scipy.sparse.linalg import spsolve, factorized, splu

from scipy.linalg import solve, lu

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from termcolor import colored, cprint
import colorama

from scipy.io import loadmat, savemat

from scipy.sparse import SparseEfficiencyWarning

warnings.simplefilter("ignore", SparseEfficiencyWarning)

from numba import jit

class TemperatureLayerClass():
    def __init__(self, zlayer, dz, c, rho, alpha, fraction, eta, beta, Cp, kappa, w, Tvec, NN, JJ, isFirst, z_start ):
        Tref = 0.0
        verbose = False

        z_start = np.int(0)

        # integer at which boundary ends.
        if (zlayer[0] == z_start ):
          if verbose: print("\nFIRST")
          print(zlayer)


        if (zlayer[0] == z_start ):
          #zlayer = zlayer - z_start
          N0 = np.int( np.round( zlayer[0] / dz ) )
          N1 = np.int( np.round( (zlayer[1] - zlayer[0] ) / dz ) )
          offset = np.int( 0 )
          if (N1+N0 > NN):
            N1 = np.int(N1-1)
            if verbose: print("LAST")
        else:
          #zlayer = zlayer - z_start
          N0 = np.int( 1 + np.round( zlayer[0] / dz ) )
          N1 = np.int( np.round( (zlayer[1] - zlayer[0] ) / dz ) )
          offset = np.int( (N0-1) * JJ )

        if (N1+N0 >= NN):
          N1 = np.int(N1-1)
          if verbose: print("LAST")
        # if (N1+N0 == NN):
        #   N0 = np.int(N0-1)

        #if (isFirst):
        if verbose:
          print("(TemperatureLayerClass) N0:", N0, ", N1:", N1, ", np.size(Tvec):", np.size(Tvec),"=", JJ*NN, ", NN(axial):", NN, ", JJ(radial):", JJ, ", zlayer:(", "{:.3f}".format(zlayer[0]), ",", "{:.3f}".format(zlayer[1]), "), N0+N1:", N1+N0 )
          print("(TemperatureLayerClass) z interval: (", "{:.3f}".format(N0*dz), ",", "{:.3f}".format(N1*dz), "), j interval: (", N0, ",", N1, "), Tvec interval: (",  np.int( (N0-1) * JJ), ",", np.int( (N0+N1) * JJ), "), zlayer: (", "{:.3f}".format(zlayer[0]), ",", "{:.3f}".format(zlayer[1]), ")" )

          print("(TemperatureLayerClass) index interval: (", "{:d}".format(N0), ",", "{:02d}".format(N1+N0), ")" )

        self.zlayer = zlayer
        if isFirst:
          self.c = c
          self.rho = rho
          self.alpha = alpha
          self.fraction = fraction
          self.eta = eta
          self.beta = beta
          self.Cp = Cp
          self.kappa = kappa
          self.w = w
        else:
          nsize = np.int( (N1+1)*JJ )
          self.c = c * np.ones((nsize,))
          self.rho = rho * np.ones((nsize,))
          self.alpha = alpha * np.ones((nsize,))
          self.fraction = fraction * np.ones((nsize,))
          self.eta = eta * np.ones((nsize,))
          self.beta = beta * np.ones((nsize,))
          self.kappa = kappa * np.ones((nsize,))
          self.w = w * np.ones((nsize,))
          self.Cp = Cp * np.ones((nsize,))

          for i in np.arange(0, nsize, dtype=np.int):
            self.Cp[i] = Cp * (1.0 + (Tvec[i+offset] - Tref) / 600.0)
            self.rho[i] = rho * (1.0 + (Tvec[i+offset] - Tref) / 600.0)
            self.w[i] = w * (1.0 + (Tvec[i+offset] - Tref) / 600.0)
            self.kappa[i] = kappa * (1.0 + (Tvec[i+offset] - Tref) / 600.0)
          if verbose:
            print("(TemperatureLayerClass) i:", i, ", offset:", offset, ", i+offset:", i+offset, ", sizes:", np.size(self.w) )
            if (N1+N0 == NN):
              print("size:", np.size(w), np.size(Tvec), "\n" )




def BHT(Grid, Layer, Q):

  #-------------------------------------------------
  # preamble, warnings and information, clear screen
  #-------------------------------------------------

  verbose = True

  willPlot = True

  colorama.init()

  # initialise timer
  t_start = time.time()

  # set heating and cooling durations:
  t_h = 0.5	# heating time (s)
  t_c = 1.5	# cooling time (s)

  # set equilibrium temperature:
  Teq = 37.0	# celcius degrees

  # Thermal damage thresholds:

  # dose (CEM43) where 0% cell necrosis occurs; safety threshold
  safety = 80.0
  # dose at which 100% cell nectosis occurs
  efficacy = 240.0
  Tbreakpoint = 43.0
  tscale = 60.0

  # temporal grid setup:
  dtmin   = 0.01
  dtscale = 100.0
  dt      = np.min([dtmin, t_h/dtscale])		# estimate timestep dt
  n       = np.ceil((t_h+t_c)/dt)	# find number of timesteps
  t       = np.linspace(0.0, t_h+t_c, np.int(n) )
  dt      = t[1]			# refined dt
  T_sp    = np.zeros((np.int(n+1),), dtype=np.float)		# spatial peak temp as a function of time
  nt      = np.int(n)

  # heat grid is four times as coarse as propagation grid in r
  rskip = 4

  # six times as coarse in z
  zskip = 6

  # new JJ
  JJ = np.size( Grid.r[0:np.size(Grid.r) :rskip] )
  # new z
  print("first z:", Grid.z[0])
  z = Grid.z[0:np.size(Grid.z) :zskip]

  # new NN
  NN = np.size( Grid.z[0:np.size(Grid.z) :zskip] )
  # new JN
  JN = JJ * NN

  # reporting:
  if (verbose):
    print('\n\tNode count')
    print('\t\tAxial\t\t%d' %NN)
    print('\t\tRadial\t\t%d' %JJ)
    print('\t\tTemporal\t%d' %nt)

  # Temp values (overwritten at each timestep
  Tvec     = np.zeros((JN,), dtype=np.float)

  # Max temp at each spatial location
  Tvec_max = np.zeros((JN,), dtype=np.float)

  # Dose values
  Dvec     = np.zeros((JN,), dtype=np.float)
  Dmat     = np.zeros((JJ, NN), dtype=np.float)

  II = np.size(Layer)

  # Get grid and operators:
  isFirst = True
  CN1, CN2, Hvec, Grid2 = BuildBHTperipherals.BuildBHTperipherals(Grid, Layer, Q, dt, verbose, rskip, zskip, Tvec, isFirst)

  # new JJ
  JJ = np.size( Grid2.r )
  # new z
  z = Grid2.z
  # new NN
  NN = np.size( Grid2.z )
  # new JN
  JN = JJ * NN

  # reporting:
  if (verbose):
    print('\n\tNode count')
    print('\t\tAxial\t\t%d' %NN)
    print('\t\tRadial\t\t%d' %JJ)
    print('\t\tTemporal\t%d' %nt)

  # new dz
  dz = Grid2.z[1] - Grid2.z[0]

  print(".....", Grid2.z[0], np.size(Grid2.z), dz, np.size(Grid2.z)*dz, z[-1], z[0], (z[-1]-z[0])/np.double(np.size(Grid2.z)) )

  dz = (z[-1]-z[0])/np.double(np.size(Grid2.z))
  print(".....",  Grid2.z[0], np.size(Grid2.z), dz, np.size(Grid2.z)*dz, z[-1], z[0], (z[-1]-z[0])/np.double(np.size(Grid2.z)) )

  # accelerate algorithm by prefactoring
  solve1 = factorized(CN1)

  if (verbose):
    string = "\titerate\ttime [s]\tMax. Heating [degrees]"
    string = string.expandtabs()
    nlen = len(string)
    print(string)
    print("-"*nlen)

  II = np.size(Layer)
  Layer[0].z = Grid.z[0]

  # Integration loop
  #-----------------
  for n in np.arange(0, nt):

    if ( t[n] < t_h ):
      #Tvec = CN1 \ ( CN2*Tvec + dt*Hvec )
      Tvec = solve1(CN2.dot(Tvec) + Hvec)
    else:
      #Tvec = CN1 \ ( CN2*Tvec )
      Tvec = solve1( CN2.dot(Tvec) )

    T_sp[n+1]   = np.max(Tvec)
    s           = np.where(Tvec > Tvec_max)
    Tvec_max[s] = Tvec[s]

    # accrue thermal dose
    if (Teq + Tvec.all() > Tbreakpoint):
      R = np.ones((JN,), dtype=np.float) / 2.0
    else:
      R = np.ones((JN,), dtype=np.float) / 4.0

    Dvec += dt * np.power( R, (Tvec+Teq - Tbreakpoint) / tscale)

    TemperatureLayer = np.ndarray((II,), dtype=np.object)

    nvec = np.zeros((II+2,), dtype=np.int)
    ivec = np.zeros((II+2,), dtype=np.int)
    nvec[-1] = NN
    ivec[-1] = NN
    for i in np.arange(0, II, dtype=np.int):
      if (i < II-1):
        nvec[i+1] = np.int( np.round((Layer[i+1].z - Layer[i].z)/dz) )
      else:
        nvec[i+1] = np.int(np.round((Grid2.z[-1] - Layer[i].z)/dz))
      ivec = np.cumsum(nvec[:i+1])
      print(i, Layer[i].z, Grid.z[-1], np.size(Layer), II, nvec[i], nvec[i] * dz, np.cumsum(nvec[:i+1]), np.cumsum(nvec[:1+i])*dz )
    ivec = np.append(ivec, NN)
    print( ivec )

    dz = Grid2.z[1] - Grid2.z[0]
    isFirst = False

    if (II==1):
      if (Layer[0].z < z[-1] ):
        zlayer = np.array((Layer[0].z, z[-1]+dz) )
        print("here 0 --", Layer[0].z, z[-1], Grid.z[-1], dz, zlayer, 1.0/dz)
      else:
        zlayer = np.array((0.0, z[-1]) )
        print("here 1")
    else:
        zlayer = np.array(( ivec[0]*dz, ivec[1]*dz -dz/2.0 ) )
        print("here 2:", dz, zlayer, z[-1]/dz, 1.0/dz)

    if ((zlayer[0] == 0) or (Layer[0].z == Grid2.z[0]) ):
      N0 = np.int( np.round( zlayer[0] / dz ) )
      N = np.int( np.round( (zlayer[1] - zlayer[0] ) / dz ) )
    else:
      N0 = np.int( 1 + np.round( zlayer[0] / dz ) )
      N = np.int( np.round( (zlayer[1] - zlayer[0] ) / dz ) )
    if (N+N0 > NN):
      N = np.int(N-1)

    #N1 = np.int(N0+N)
    i=0
    #print("i: ", i, ",\tzlayer: ", np.round(zlayer,2), ",\tLayer.z: ", np.round(Layer[i].z,2), ",\tN0: ", N0, ",\tN1: ", N1, ",\tz[N0]: ", np.round(z[N0],3), ",\tz[N1]: ", np.round(z[N1],3), ",\tz[end]: ", z[-1] )
    if (II==1):
      TemperatureLayer[0] = TemperatureLayerClass(zlayer, dz, 1482.0, 1000.0, 0.217, 0.0, 2.0, 3.5, 4180.0, 0.6, 0.0, Tvec, NN, JJ, isFirst, Grid2.z[0])
    else:
      print("----->", zlayer)
      TemperatureLayer[0] = TemperatureLayerClass(zlayer, dz, 1482.0, 1000.0, 0.217, 0.0, 2.0, 3.5, 4180.0, 0.6, 0.0, Tvec, NN, JJ, isFirst, Grid2.z[0])

    if (II > 1):
      for i in np.arange(1,II, dtype=np.int):
        if (i == 0):
          if ( np.abs(Layer[i].z - Grid2.z[0]) < 10E-3 ):
            zlayer = np.array((Grid2.z[0], z[-1]+0.01))
            print("here a")
          else:
            zlayer = np.array((Layer[i-1].z, Layer[i].z))
            zlayer = np.array(( ivec[i]*dz, ivec[i+1]*dz -dz/2.0 ) )
            print("here b")
        if (i==II-1):
          zlayer = np.array((Layer[i].z, z[-1] ))
          zlayer = np.array(( ivec[i]*dz, ivec[i+1]*dz -dz/2.0 ) )
          print("here c")
        else:
          zlayer = np.array((Layer[i].z, Layer[i+1].z-dz/2.0))
          zlayer = np.array(( ivec[i]*dz, ivec[i+1]*dz -dz/2.0 ) )
          print("here d")

        # N0 = np.int( np.round( zlayer[0] / dz ) )
        # N = np.int( np.round( (zlayer[1] - zlayer[0] ) / dz ) )
        # if (N+N0 > NN):
        #   N = np.int(N-1)

        # N1 = np.int(N0+N)

        #print(i, np.round(zlayer,2), np.round(Layer[i].z,2), N0, N1, np.round(N0*dz,2), np.round(N1*dz,2), np.round(z[N0],3), np.round(z[N1-1],3) )
        #print("i: ", i, ",\tzlayer: ", np.round(zlayer,2), ",\tLayer.z: ", np.round(Layer[i].z,2), ",\tN0: ", N0, ",\tN1: ", N1, ",\tz[N0]: ", np.round(z[N0],3), ",\tz[N1]: ", np.round(z[N1],3), ",\tz[end]: ", z[-1] )
        print("----->", zlayer)
        TemperatureLayer[i] = TemperatureLayerClass(zlayer, dz, 1482.0, 1000.0, 0.217, 0.0, 2.0, 3.5, 4180.0, 0.6, 0.0, Tvec, NN, JJ, isFirst, Grid2.z[0])

    # zlayer = np.array((4.0001, 6.01))
    #TemperatureLayer[1] = TemperatureLayerClass(zlayer, dz, 1482.0, 1000.0, 2.17, 0.0, 2.0, 3.5, 4180.0, 0.6, 0.0, Tvec, NN, JJ, isFirst, Grid2.z[0])

    #zlayer = np.array((Layer[-1].z+0.01, Grid2.z[-1]+0.01))
    #TemperatureLayer[np.size(Layer)] = TemperatureLayerClass(zlayer, dz, 1482.0, 1000.0, 0.217, 0.0, 2.0, 3.5, 4180.0, 0.6, 0.0, Tvec, NN, JJ, isFirst, Grid2.z[0])

    # zlayer = np.array((6.0, Grid2.z[-1]))
    # TemperatureLayer[3] = TemperatureLayerClass(zlayer, dz, 1482.0, 1000.0, 0.217, 0.0, 2.0, 3.5, 4180.0, 0.6, 0.0, Tvec, NN, JJ, isFirst, Grid2.z[0])

    #CN1, CN2, Hvec, Grid2 = BuildBHTperipherals.BuildBHTperipheralsTemperature(Grid2, TemperatureLayer, Q, dt, rskip, zskip, verbose, Tvec, isFirst)

    if (verbose):
      if ((n != 0 ) and (np.mod(n, 10) == 0) ):
        print("\t%3d\t%3.2f\t\t%6.4e" %(n, t[n], T_sp[n+1]))
      if ((n == nt-1) and (np.mod(n, 10) != 0) ) :
        print("\t%3d\t%3.2f\t\t%6.4e\n" %(n, t[n], T_sp[n+1]))

  #-------------------------------------------------
  elapsed = time.time() - t_start
  hours = np.floor(elapsed/3600.0)
  minutes = np.floor( (elapsed - hours*3600.0)/60.0 )
  seconds = elapsed - 60.0*(hours*60.0 + minutes)
  printfin = colored('... finished', 'blue')
  if (hours >= 1):
    if (hours >1):
      if (minutes > 1):
        print( printfin + ' in {0} hours, {1} minutes and {2} seconds\n'.format(int(hours), int(minutes), seconds) )
      if (minutes == 1):
        print( printfin + ' in {0} hours, {1} minute and {2} seconds\n'.format(int(hours), int(minutes), seconds) )
      if (minutes < 1):
        print( printfin + ' in {0} hours and {1} seconds'.format(int(hours), seconds) )
    else:
      if (minutes > 1):
        print( printfin + ' in {0} hour, {1} minutes and {2:.2%} seconds\n'.format(int(hours), int(minutes), seconds) )
      if (minutes == 1):
        print( printfin + ' in {0} hour, {1} minute and {2:.2%} seconds\n'.format(int(hours), int(minutes), seconds) )
      if (minutes < 1):
        print( printfin + ' in {0} hour and {1} seconds\n'.format(int(hours), seconds) )
  if ((hours < 1) and (minutes >= 1)):
    if (minutes == 1):
      print( printfin + ' in {0} minute and {1} seconds\n'.format(int(minutes), seconds) )
    else:
      print( printfin + ' in {0} minutes and {1:0.2f} seconds\n'.format(int(minutes), seconds) )
  if ((hours < 1) and (minutes < 1)):
    print( printfin + " in %5.3f seconds.\n" %(seconds) )


  if (willPlot):
    # convert vectors to matrices for presentation
    r    = np.hstack( (-Grid2.r[Grid2.JJ-1:0: -1], Grid2.r) )

    Dmat = np.reshape( Dvec, (Grid2.JJ,Grid2.NN) )
    Dmat = np.vstack( (Dmat[Grid2.JJ-1:0: -1,:], Dmat) )

    Tmat_max = np.reshape( Tvec_max+Teq, (Grid2.JJ,Grid2.NN) )
    Tmat_max = np.vstack( (Tmat_max[Grid2.JJ-1:0: -1,:], Tmat_max) )

    #-------------------------------------------------

    # specify whether colours for figures are defined as rgb or cmyk
    colorscheme=1
    # define dictionary of colours
    if colorscheme==1:
      ICRcolors = { \
      'ICRgray': (98.0/255.0, 100.0/255.0, 102.0/255.0), \
      'ICRgreen':(202.0/255.0, 222.0/255.0, 2.0/255.0), \
      'ICRred': (166.0/255.0, 25.0/255.0, 48.0/255.0), \
      'ICRpink': (237.0/255.0, 126.0/255.0, 166.0/255.0), \
      'ICRorange': (250.0/255.0, 162.0/255.0, 0.0/255.0), \
      'ICRyellow': (255.0/255.0, 82.0/255.0, 207.0/255.0), \
      'ICRolive': (78.0/255.0, 89.0/255.0, 7.0/255.0), \
      'ICRdamson': (79.0/255.0, 3.0/255.0, 65.0/255.0), \
      'ICRbrightred': (255.0/255.0, 15.0/255.0, 56.0/255.0), \
      'ICRlightgray': (59.0/255.0, 168.0/255.0, 170.0/255.0), \
      'ICRblue': (0.0/255.0, 51.0/255.0, 41.0/255.0)}
    else:
      ICRcolors = { \
      'ICRgray': (0.04, 0.02, 0.00, 0.60), \
      'ICRgreen': (0.09, 0.00, 0.99, 0.13), \
      'ICRred': (0.00, 0.85, 0.71, 0.35), \
      'ICRpink': (0.00, 0.47, 0.30, 0.07), \
      'ICRorange': (0.00, 0.35, 1.00, 0.02), \
      'ICRyellow': (0.00, 0.68, 0.19, 0.00), \
      'ICRolive': (0.24, 0.13, 0.93, 0.60), \
      'ICRdamson': (0.21, 0.97, 0.35, 0.61), \
      'ICRbrightred': (0.00, 0.94, 0.78, 0.00), \
      'ICRlightgray': (0.16, 0.11, 0.10, 0.26), \
      'ICRblue': (1.00, 0.00, 0.20, 0.80) }

    # render text with TeX
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    xlabelfontsize = 14
    ylabelfontsize = 14
    titlefontsize = 14
    xlabelcolour = 'black'
    ylabelcolour = 'black'
    titlecolour = 'black'

    # define figure and axis handles
    _, ax1 = plt.subplots()
    # plot peak temperatures
    ax1.plot( t, T_sp[0:np.shape(t)[0]], marker='o', color=ICRcolors['ICRpink'] )
    # define xticks
    xticks = np.arange( t[0],t[-1], 10)
    # define xlabel
    ax1.set_xlabel(r'$t$ [sec]', fontsize=xlabelfontsize, color=xlabelcolour)
    # define ylabel
    ax1.set_ylabel(r'$\Delta T$ [Degrees ${}^{\circ}$C]', fontsize=ylabelfontsize, color=ylabelcolour)
    # set title
    ax1.set_title(r'Enhanced Peak Temperature', fontsize=titlefontsize, color=titlecolour)
    # apply xticks
    ax1.set_xticks(xticks,minor=True)
    # set minor ticks on both axes
    ax1.minorticks_on()
    # apply grid to figure
    plt.grid(True)

    fig2, ax2 = plt.subplots()
    cDmat = ax2.contourf(Grid2.z, r, Dmat)
    if ( np.amax(Dmat) > efficacy ):
      ax2.contour( Grid2.z, r, Dmat, levels=[safety, efficacy], linewidths=2)
    if ( (np.amax(Dmat) > safety) and (np.amax(Dmat) < efficacy) ):
      ax2.contour( Grid2.z, r, Dmat, levels=[safety], linewidths=2)
    ax2.set_xlabel(r'$z$ [cm]', fontsize=xlabelfontsize, color=xlabelcolour)
    ax2.set_ylabel(r'$r$ [cm]', fontsize=ylabelfontsize, color=ylabelcolour)
    ax2.set_title(r'Thermal dose accumulation', fontsize=titlefontsize, color=titlecolour)
    fig2.colorbar(cDmat, ax=ax2)
    plt.grid(True)

    fig3, ax3 = plt.subplots()
    cTmax = ax3.contourf(Grid2.z, r, Tmat_max)
    ax3.set_xlabel(r'$z$ [cm]', fontsize=xlabelfontsize, color=xlabelcolour)
    ax3.set_ylabel(r'$r$ [cm]', fontsize=ylabelfontsize, color=ylabelcolour)
    ax3.set_title(r'Maximum Temperatures', fontsize=titlefontsize, color=titlecolour)
    fig3.colorbar(cTmax, ax=ax3)
    plt.grid(True)

    #-------------------------------------------------

    # render figures
    plt.show()
