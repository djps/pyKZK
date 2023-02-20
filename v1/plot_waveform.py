def plot_waveform(p0,f,X,z,K):

  """

  plots a single cycle of the time-domain waveform

  """

  import numpy as np

  import matplotlib.pyplot as plt
  import matplotlib.mlab as mlab

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

  # location of peak intensity
  zpoint = 0.01*np.around(100*z)

  # time step
  dtt = 1.0/f/(K-1)
  
  # scaled waveform
  XX = 1e-6*p0*np.squeeze(X)
  
  # length
  NN = np.shape(X)[0]

  # time domain 
  tt = np.linspace(0, 1e6/f, NN)
  
  # render text with TeX
  plt.rc('text', usetex=True)
  plt.rc('font', family='serif')
  
  # define figure and axis handles
  fig1, ax = plt.subplots()
  
  # hold on for multiple plots on figure
  fig1.hold(True)

  # plot peak temperature
  ax.plot( np.squeeze(tt[0:N]), np.squeeze(np.transpose(XX[0:N])), linewidth=2, linestyle='-', color=ICRcolors['ICRred'] )

  # define xticks
  xticks = np.arange(0,tt[-1],10)

  # define xlabel
  ax.set_xlabel(r't [$\mu$ \mathrm{sec}]', fontsize=14, color='black')

  # define ylabel
  ax.set_ylabel(r'p [\mathrm{MPa}]', fontsize=14, color='black')

  # set title
  ax.set_title(r'\mathrm{Waveform at }z = ', str(zpoint), r' \mathrm{cm}'], fontsize=14, color='black')

  # apply xticks
  ax.set_xticks(xticks,minor=True)

  # apply grid to figure
  plt.grid(True)

  # render figure
  plt.show()

  # delete variables to clear workspace memory
  del dtt, XX, NN, tt, zpoint