def axisymmetricKZK(drive,efficiency):

  """

  Driver for axisymmetric KZK integrator.

  """

  import numpy as np
  import time, os, warnings

  from math import pi
  from scipy.io import loadmat, savemat
  from scipy.sparse.linalg import spsolve, factorized
  from scipy.linalg import solve, lu

  import equivalent_time, timing, KZK_parameters, computational_grid, initial_condition, KZK_operators, TDNL

  # get system parameters
  p0,c1,c2,rho1,rho2,N1,N2,G1,G2,gamma1,gamma2,alpha1,alpha2,a,b,d,f,R,Z,z_,K = KZK_parameters.KZK_parameters(drive,efficiency)

  # set size of arrays which store complex values for the K harmonics
  K2 = 2*K

  # print coefficients to screen
  print '\n\tp0 = %2.2f MPa' %(1e-6*p0)
  print '\tN1 = %1.2f\tN2 = %1.2f' %(N1,N2)
  print '\tG1 = %3.2f\tG2 = %3.2f\n' %(G1,G2)

  # check F-number satisfies paraxial criteria
  F = 0.5*d/a
  if ( F < 1.37):
    print ' *** Warning -- f/%1.2f exceeds the conditions under which KZK is derived (> f/1.37) *** \n' %(F)

  # grid set-up
  M,J,J_,dz,dr,z,r = computational_grid.computational_grid(Z,R,max(G1,G2),a,d,gamma2[0],N2)

  # dependent variables u, v
  u = np.zeros((2*J,K))

  limit = 1.0/np.sqrt( 1.0 - (a/d)**2 )
  v = initial_condition.initial_condition(J,K,G1,r,b*limit/a,limit)
  v[0:J,0]   = v[0:J,0]*np.sqrt( 1.0-(r/d)**2 )
  v[J:2*J,0] = v[J:2*J,0]*np.sqrt( 1.0 - (r/d)**2 ) # why this is done over two lines i don't know!

  op = [KZK_operators.KZK_operators(r,R,G1,dz,dr,J,k,gamma1[k]) for k in np.arange(0,K)]
  IRK11, IRK12, CN11, CN12 = zip(*op)

  op = [KZK_operators.KZK_operators(r,R,G1,dz,dr,J,k,gamma2[k]) for k in np.arange(0,K)]
  IRK21, IRK22, CN21, CN22 = zip(*op)

  #print np.shape(rmat), np.shape(r)

  #rmat = np.squeeze(rmat)

  #print r[0], rmat[0], np.shape(r), np.shape(rmat), r[-1], rmat[-1]

  #print np.max( np.abs( rmat - r ) ), '\n'

  #print gamma1, '\n'

  ##print type(test1), type(IRK11[1]), type(test2), type(IRK12[1]), '\n'
  ##print '\n', np.shape(test1), np.shape(test2), np.shape(IRK11[1]), np.shape(IRK12[1]), '\n'
  #print "=>", test1[0,0],  1.0*IRK11[1][0,0], "<=", '\n'
  ##print "=>", test2[0,0],  1.0*IRK12[1][0,0], "<=", '\n'
  #print "\ttest1", test1[0:15,0:15], '\n'
  #print "\tIRK11",  1.0*IRK11[1][0:15,0:15], '\n'
  ##print "\ttest2", test2[0:15,0:15], '\n'
  ##print "\tIRK12",  1.0*IRK12[1][0:15,0:15], '\n'
  ##print np.shape(test1.data),np.shape(IRK11[1].data), np.shape(test2.data),np.shape(IRK12[1].data), '\n'
  ##print np.shape(test1.indices), np.shape(IRK11[1].indices), np.shape(test2.indices), np.shape(IRK12[1].indices), '\n'
  ##print np.shape(test1.indptr), np.shape(IRK11[1].indptr), np.shape(test2.indptr), np.shape(IRK12[1].indptr), '\n'
  #print np.max( np.max( np.abs( test1 -  1.0*IRK11[1] ) ) ), '\n'
  ##print np.max( np.max( np.abs( test2 -  1.0*IRK12[1] ) ) ), '\n'
  #print "***", test1[0,J], test1[0,J+1], 1.0*IRK11[1][0,J], 1.0*IRK11[1][0,J+1], "***", '\n'
  ##print "***", test2[0,J], test2[0,J+1], 1.0*IRK12[1][0,J], 1.0*IRK12[1][0,J+1], "***", '\n'

  #print "----------------------------------------------------"

  #print type(test1), type(IRK11[1]), type(test2), type(IRK12[1]), '\n'
  #print '\n', np.shape(test1), np.shape(test2), np.shape(IRK11[1]), np.shape(IRK12[1]), '\n'
  #print "=>", test1[0,0],  1.0*IRK11[1][0,0], "<=", '\n'
  #print "=>", testmat2[0,0], 1.0*IRK12[1][0,0], "<=", '\n'
  #print "\ttest1", test1[0:15,0:15], '\n'
  #print "\tIRK11",  1.0*IRK11[1][0:15,0:15], '\n'
  #print "\ttestmat2",'\n', testmat2[:,0], '\n'
  #print "\tIRK12", '\n', 1.0*IRK12[1][:,0], '\n'
  #print np.shape(test1.data),np.shape(IRK11[1].data), np.shape(test2.data),np.shape(IRK12[1].data), '\n'
  #print np.shape(test1.indices), np.shape(IRK11[1].indices), np.shape(test2.indices), np.shape(IRK12[1].indices), '\n'
  #print np.shape(test1.indptr), np.shape(IRK11[1].indptr), np.shape(test2.indptr), np.shape(IRK12[1].indptr), '\n'
  #print np.max( np.max( np.abs( test1 -  1.0*IRK11[1] ) ) ), '\n'
  #print "\tdiff", '\n', np.max( np.max( np.abs( testmat2[:,0] -  1.0*IRK12[1][:,0] ) ) ), '\n'
  #print "***", test1[0,J], test1[0,J+1], 1.0*IRK11[1][0,J], 1.0*IRK11[1][0,J+1], "***", '\n'
  #print "***", testmat2[0,J], testmat2[0,J+1], 1.0*IRK12[1][0,J], 1.0*IRK12[1][0,J+1], "***", '\n'

  # IRK slope vectors
  k1 = np.zeros( (2*J,1) )
  k2 = np.zeros( (2*J,1) )

  # IRK coefficients
  b1 = 1.0/np.sqrt(2.0)
  b2 = 1.0 - b1

  # parameters for nonlinear integration
  #-------------------------------------

  # nonlinear term integration parameters
  mu1 = N1*K*dz/pi
  mu2 = N2*K*dz/pi

  # cutoffs for performing nonlinear integration
  cutoff1 = gamma1[0]/10.0/N1
  cutoff2 = gamma2[0]/10.0/N2

  # data vectors
  X = np.zeros((1,K2)) #, dtype=complex)
  Y = np.zeros((1,K2), dtype=complex)

  #X = np.zeros((1,K2))
  #Y = np.zeros((1,K2))

  # waveform at position of maximum intensity
  Xpeak = np.zeros( (1,K2) )

  # for plotting routines:
  #-------------------------------------

  # Heating rate matrices
  H  = np.zeros((J_,M+1))
  H2 = np.zeros((J_,M+1))
  H[:,0] = np.real( gamma1[0] )*( v[0:J_,0]**2 + v[J:J+J_,0]**2 )
  I = np.zeros((J_,M+1))
  I[:,0] = v[0:J_,0]**2 + v[J:J+J_,0]**2
  I_td = np.zeros((J,1))
  dt = 1.0/f/(K2-1.0)

  # axial intensity vector
  Ix = np.zeros((M+1,1))
  Ix[0] = v[0,0]**2 + v[J,0]**2

  # radial intensity vector
  Ir = np.zeros((J_,1))

  # first (up to) 5 harmonic pressure amplitudes in axial direction
  if (K<5):
    kk = K
  else:
    kk = 5
  p5x = np.zeros((kk,M+1))
  p5x[0,0] = np.sqrt( v[0,0]*v[0,0] + v[J,0]*v[J,0] )

  # first (up to) 5 harmonic pressure amplitudes in radial direction
  p5r = np.zeros((kk,J_))

  # axial peak positive pressures
  peak = np.zeros((M+1,1))

  # axial peak negative pressures
  trough = np.zeros((M+1,1))
  peak[0]   = 1.0
  trough[0] =-1.0

  # for monitoring simulation:
  #-------------------------------------

  # amplitude of Kth harmonic
  amp_K = 0

  # for determining peak pressure waveform and location of its occurence:
  #-------------------------------------

  # index of last meshpoint in material 1? ceil rounds up
  m_t = np.ceil(z_/dz/d)
  if(m_t>M):
    m_t = M

  # index of meshpoint nearest focus
  m_f = np.around(M/Z)

  # peak axial pressure
  p_peak = 0.0

  # distance at which peak axial pressure occurs
  z_peak = 0.0

  # allocate memory
  Ppos = np.zeros((J,M))
  Pneg = np.zeros((J,M))

  # integrate the equations through material 1
  print 'Integrating KZK equation...\n'
  print '\tz (cm)\t\ttime (hr:min:sec) \tn'

  # start timer
  steps = np.floor( np.linspace(0,M-1,num=10) )
  steps = steps.astype(int)

  t_start = time.time()
  p1 = 0

  #--------------------------------------------------------------------------------------------------

  # spatial front: from 0 to m_t-1
  for m in np.arange(0,m_t):

    # compute updated v
    v,X,Ppos[:,m],Pneg[:,m],I_td = TDNL.TDNL(v,X,Y,K,J,mu1,cutoff1,Ppos[:,m],Pneg[:,m],I_td)

    # update time-domain intensity
    I_td = f*dt*I_td/dz

    # update peak positive pressure
    peak[m+1]   = np.max(np.real(X))

    # update peak negative pressure
    trough[m+1] = np.min(np.real(X))

    # if at focus get radial information
    if (m==m_f):
      # radial intensity as sum of harmonics
      if (K>1):
	Ir = np.sum( np.transpose( v[0:J_,:] )**2 )
	Ir = Ir + np.sum( np.transpose( v[J:J+J_,:] )**2 )
      else:
	Ir = v[0:J_]**2 + v[J:J+J_]**2
      # first five harmonics
      p5r = np.sqrt( v[0:J_,0:kk]**2 + v[J:J+J_,0:kk]**2 )

    # peak is maximal get updated data
    if (peak[m+1]>p_peak):
      p_peak = peak[m+1]
      z_peak = z[m+1]
      Xpeak  = X

    # if in the near field solve with an implicit Runge-Kutta scheme, else using a Crank-Nicolson scheme
    if (z[m]<0.3):

      # for each harmonic
      for k in np.arange(0,K):

	#k1 = spsolve(IRK11[k], IRK12[k]*v[:,k] )
	k1 = IRK11[k]( IRK12[k]*v[:,k] )
	#k2 = spsolve(IRK11[k], IRK22[k]*(v[:,k] + dz*b1*k1) )
	k2 = IRK11[k](IRK22[k]*(v[:,k] + dz*b1*k1) )

	# update acoustic field
	u[:,k] = v[:,k] + dz*( b1*k1 + b2*k2 )

    else:
      for k in np.arange(0,K):
	u[:,k] = spsolve(CN11[k], CN12[k]*v[:,k] )

    # reassign working variable
    v = u

    # reassign amplitude of K-th harmonic
    pl = np.sqrt( v[0,K-1]**2 + v[J,K-1]**2 )
    if (pl > amp_K):
      amp_K = pl

    # update heating rates
    for j in np.arange(0,J_):
      H[j,m+1]  = np.sum( np.real( np.transpose(gamma1[:]) )*( u[j,:]**2 + u[J+j,:]**2 ) )
      H2[j,m+1] = I_td[j]
      I[j,m+1]  = np.sum( u[j,:]**2 + u[J+j,:]**2 )

    # update axial Intensity
    Ix[m+1] = np.sum( v[0,:]**2 + v[J+1,:]**2 )

    # gain check
    if ( Ix[m+1] > 2*G1*G1 ):
      print '\tStopped - computation became unstable at z = %2.1f cm.\n' %(d*z[m])
      r = a*r[0:J_]
      z = d*z
      #KZK_radial_plots.KZK_radial_plots(r,Ir,H(:,round((M+1)/Z)),p5r,p0,rho2,c2,R,a);
      #KZK_axial_plots.KZK_axial_plots(z,Ix,p5x,H(1,:),peak,trough,p0,rho1,rho2,c1,c2,d,Z,a,m_t)
      exit(0)

    # axial pressure values for first five harmonics
    p5x[:,m+1] = np.sqrt( v[0,0:kk]*v[0,0:kk] + v[J,0:kk]*v[J,0:kk] )

    # timing routine
    if m in steps:
      p2 = np.floor(10*(m+1)/M)
      p1 = timing.timing(p1,p2,t_start,z[m],d,m)

  # material 2:
  #------------------------------------------------------------------------------------------------

  # impediance mismatch loss
  v = 2.0*rho2*c2*v/(rho1*c1+rho2*c2)

  for m in np.arange(m_t,M):
    v,X,Ppos[:,m],Pneg[:,m],I_td = TDNL.TDNL(v,X,Y,K,J,mu2,cutoff2,Ppos[:,m],Pneg[:,m],I_td)
    I_td = f*dt*I_td/dz
    peak[m+1]   = np.max(np.real(X))
    trough[m+1] = np.min(np.real(X))

    if (m==m_f):
      if(K>1):
	Ir = sum( ( np.transpose(v[0:J_,:]) )**2 )
	Ir = Ir + sum( ( np.transpose(v[J:J+J_,:]))**2)
      else:
	Ir = v[0:J_]**2 + v[J:J+J_]**2

      p5r = np.sqrt( v[0:J_,0:kk]**2 + v[J:J+J_,0:kk]**2 )

    if (peak[m+1]>p_peak):
      p_peak = peak[m+1]
      z_peak = z[m+1]
      Xpeak = X

    if (z[m]<0.3):

      if (m==m_t):

	for k in np.arange(0,K):
	  #k1 = spsolve(IRK11[k], IRK12[k]*v[:,k] )
	  #k2 = spsolve(IRK21[k], IRK22[k]*(v[:,k] + dz*b1*k1) )

	  #k1 = spsolve(IRK11[k], IRK12[k]*v[:,k] )
	  k1 = IRK11[k]( IRK12[k]*v[:,k] )
	  #k2 = spsolve(IRK11[k], IRK22[k]*(v[:,k] + dz*b1*k1) )
	  k2 = IRK21[k](IRK22[k]*(v[:,k] + dz*b1*k1) )

	  u[:,k] = v[:,k] + dz*(b1*k1 + b2*k2)

	u = (1.0 - (rho1*c1-rho2*c2)/(rho1*c1+rho2*c2))*u

      else:

	for k in np.arange(0,K):
	  #k1 = spsolve(IRK21[k], IRK22[k]*v[:,k])
	  k1 = IRK21[k]( IRK22[k]*v[:,k] )
	  #k2 = spsolve(IRK21[k], IRK22[k]*(v[:,k]+dz*b1*k1))
	  k2 = IRK21[k](IRK22[k]*(v[:,k]+dz*b1*k1))
	  u[:,k] = v[:,k] + dz*(b1*k1 + b2*k2)

    else:

      for k in np.arange(0,K):
	u[:,k] = spsolve(CN21[k], CN22[k]*v[:,k])

    v = u
    pl = np.sqrt( v[0,K-1]**2 + v[J,K-1]**2)

    if (pl > amp_K):
      amp_K = pl

    for j in np.arange(0,J_):
      H[j,m+1]  = sum( np.real( np.transpose(gamma2[:]) )*( u[j,:]**2 + u[J+j,:]**2) )
      H2[j,m+1] = I_td[j]
      I[j,m+1]  = sum( u[j,:]**2 + u[J+j,:]**2 )

    Ix[m+1] = sum( v[0,:]**2 + v[J,:]**2 )

    if ( Ix[m+1] > 2*G2*G2 ):
      print '\tStopped - computation became unstable at z = %2.1f cm.\n' %(d*z[m])
      r = a*r[0:J_]
      z = d*z
      #KZK_radial_plots.KZK_radial_plots(r,Ir,H(:,round((M+1)/Z)),p5r,p0,rho2,c2,R,a);
      #KZK_axial_plots.KZK_axial_plots(z,Ix,p5x,H[1,:],peak,trough,p0,rho1,rho2,c1,c2,d,Z,a,m_t)
      exit(0)

    p5x[:,m+1] = np.sqrt( v[0,0:kk]*v[0,0:kk] + v[J+1,0:kk]*v[J+1,0:kk] )

    if m in steps:
      p2 = np.floor(10*(m+1)/M)
      p1 = timing.timing(p1,p2,t_start,z[m],d,m)

  # dimensionalize H
  H[:,0:m_t+1]  = 1e-4*p0*p0*H[:,0:m_t+1]/rho1/c1/d
  H[:,m_t+1:M]  = 1e-4*p0*p0*H[:,m_t+1:M]/rho2/c2/d

  # dimensionalize H2
  H2[:,0:m_t+1] = 1e-4*0.5*p0*p0*H2[:,0:m_t+1]/rho1/c1/d
  H2[:,m_t+1:M] = 1e-4*0.5*p0*p0*H2[:,m_t+1:M]/rho2/c2/d

  # dimensionalize I
  I[:,0:m_t+1] = 1e-4*0.5*p0*p0*I[:,0:m_t+1]/rho1/c1
  I[:,m_t+1:M] = 1e-4*0.5*p0*p0*I[:,m_t+1:M]/rho2/c2

  H = np.real(H + H2)

  print '\n\tmax(|p_'+str(K)+'|/p0) = %e \n' %(amp_K)
  print '\tratio 2nd harmonic to 1st   = %6.5f \n' %( max( np.max(p5x[1,:]), np.max(p5r[:,1]))/max(np.max(p5x[0,:]), np.max(p5r[:,0])) )
  print '\tratio 3rd harmonic to 1st   = %6.5f \n'   %( max(np.max(p5x[2,:]),np.max(p5r[:,2]))/max(np.max(p5x[0,:]),np.max(p5r[:,0])) )
  print '\tratio 1st harmonic to total = %6.5f \n'   %( max(np.max(p5x[0,:]),np.max(p5r[:,0]))/np.max(peak) )
  print '\tratio 2nd harmonic to total = %6.5f \n'   %( max(np.max(p5x[1,:]),np.max(p5r[:,1]))/np.max(peak) )

  # rescale r and chop so that PML region is excluded in plots
  r = a*r[0:J_]

  # rescale z
  z = d*z

  # peak positive
  Ppos = Ppos[0:J_,:]

  # peak negative
  Pneg = Pneg[0:J_,:]

  del H2,amp_K

  #print "\n"

  ##--------------------------------------------------------------------------------------------------
  #ifig=1

  #if (ifig==1):
    #  import matplotlib.pyplot as plt
    #  import matplotlib.mlab as mlab

    ## define colours for figures
    #colorscheme=1
    #if colorscheme==1:
      #ICRcolors = { \
      #'ICRgray': (98.0/255.0, 100.0/255.0, 102.0/255.0), \
      #'ICRgreen':(202.0/255.0, 222.0/255.0, 2.0/255.0), \
      #'ICRred': (166.0/255.0, 25.0/255.0, 48.0/255.0), \
      #'ICRpink': (237.0/255.0, 126.0/255.0, 166.0/255.0), \
      #'ICRorange': (250.0/255.0, 162.0/255.0, 0.0/255.0), \
      #'ICRyellow': (255.0/255.0, 82.0/255.0, 207.0/255.0), \
      #'ICRolive': (78.0/255.0, 89.0/255.0, 7.0/255.0), \
      #'ICRdamson': (79.0/255.0, 3.0/255.0, 65.0/255.0), \
      #'ICRbrightred': (255.0/255.0, 15.0/255.0, 56.0/255.0), \
      #'ICRlightgray': (59.0/255.0, 168.0/255.0, 170.0/255.0), \
      #'ICRblue': (0.0/255.0, 51.0/255.0, 41.0/255.0)}
    #else:
      #ICRcolors = { \
      #'ICRgray': (0.04, 0.02, 0.00, 0.60), \
      #'ICRgreen': (0.09, 0.00, 0.99, 0.13), \
      #'ICRred': (0.00, 0.85, 0.71, 0.35), \
      #'ICRpink': (0.00, 0.47, 0.30, 0.07), \
      #'ICRorange': (0.00, 0.35, 1.00, 0.02), \
      #'ICRyellow': (0.00, 0.68, 0.19, 0.00), \
      #'ICRolive': (0.24, 0.13, 0.93, 0.60), \
      #'ICRdamson': (0.21, 0.97, 0.35, 0.61), \
      #'ICRbrightred': (0.00, 0.94, 0.78, 0.00), \
      #'ICRlightgray': (0.16, 0.11, 0.10, 0.26), \
      #'ICRblue': (1.00, 0.00, 0.20, 0.80) }
    ## safety check for length of vectors for plotting
    #Nlength = np.min( np.array([np.shape(np.squeeze(z))[0], np.shape(np.squeeze(trough))[0], np.shape(np.squeeze(peak))[0]]) )
    ## render text with TeX
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    #fig1, ax = plt.subplots()
    #fig1.hold(True)
    #ax.set_xlim([0,Z*d])
    #xticks = np.arange(0,Z*d,10)
    #ax.set_xlabel(r'z [cm]', fontsize=14, color='black')
    #ax.set_ylabel(r'Pressure [MPa]', fontsize=14, color='black')
    #ax.set_title(r'Axial Pressures', fontsize=14, color='black')
    #ax.set_xticks(xticks,minor=True)
    #ax.minorticks_on()
    #ax.plot( np.squeeze(z)[0:Nlength], np.squeeze(np.transpose(peak))[0:Nlength], linewidth=2, linestyle='-', color=ICRcolors['ICRred'] )
    #ax.plot( np.squeeze(z)[0:Nlength], np.squeeze(np.transpose(trough))[0:Nlength], linewidth=2, linestyle='-', color=ICRcolors['ICRgreen'] )
    #plt.grid(True)
    #plt.show()

  return z,r,H,I,Ppos,Pneg,Ix,Ir,p0,p5r,p5x,peak,trough,rho1,rho2,c1,c2,R,d,Z,M,a,m_t,f,Xpeak,z_peak,K2,z_
