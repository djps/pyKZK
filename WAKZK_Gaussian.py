# load packages

import os, time, datetime, warnings
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import axes3d, Axes3D

from math import pi, log

from termcolor import colored, cprint

from scipy.io import loadmat, savemat
from scipy.sparse import diags, spdiags, eye, bmat
from scipy.sparse.linalg import splu, factorized

from scipy.sparse.linalg import spsolve, factorized, splu
from scipy.linalg import solve, lu

from scipy.io import loadmat, savemat

from scipy.sparse import SparseEfficiencyWarning

import timeit

from pprint import pprint

import SourceFilterH, SynthAxScan, SynthRadScan, BuildPade11operators, BuildPade12operators, TDNL

# suppress warnings
warnings.simplefilter("ignore", SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class LayerClass():
    def __init__(self, z, c, rho, alpha, fraction, eta, beta, Cp, kappa, w, index):
        self.z = z
        self.c = c
        self.rho = rho
        self.alpha = alpha
        self.fraction = fraction
        self.eta = eta
        self.beta = beta
        self.Cp = Cp
        self.kappa = kappa
        self.w = w
        self.index = index

class TransducerClass():
    def __init__(self, f, a1, a2, d, P):
        self.f = f
        self.a1 = a1
        self.a2 = a2
        self.d = d
        self.P = P

class GridClass():
    def __init__(self, Z, KK, R, JJ, NN, r, z):
        self.Z = Z
        self.KK = np.int(KK)
        self.R = R
        self.JJ = np.int(JJ)
        self.NN = np.int(NN)
        self.r = r
        self.z = z

class SpecOutClass():
    def __init__(self, w, p_r, p_c, p5, I):
        self.w  = w
        self.p_r = p_r
        self.p_c = p_c
        self.p5 = p5
        self.I = I


def WAKZK_Gaussian(willPlot = False):

    '''
    Implementation of wide-angle parabolic method for axisymmetric HITU beams.
    '''


    output = False

    verbose = False

    debug = False

    tstart = timeit.default_timer()

    minharmonics = 5

    ## Transducer ##############################################################

    f = 1.0E6
    a1 = 0.0
    a2 = 1.0
    d = 5.0
    P = 100
    Tx = TransducerClass(f, a1, a2, d, P)


    ## Spatial averaging ######################################################

    # diameter of hydrophone element (mm).  [hd = 0 no averaging]
    hd = 0.0


    ## Layered media ###########################################################

    # number of layers
    II = 2

    Layer = np.ndarray((II,), dtype=np.object)
    #                    ( z,  c,      rho,    alpha, fraction, eta, beta, Cp, kappa, w, index)
    Layer[0] = LayerClass(0.0, 1482.0, 1000.0, 0.217, 0.1, 2.0, 3.5, 4180.0, 0.6, 0.0,  0)
    Layer[1] = LayerClass(3.0, 1629.0, 1000.0, 58.0,  0.9, 1.0, 4.5, 4180.0, 0.6, 20.0, 0)
    # dummy layer
    #Layer[2] = LayerClass(6.0, 1482.0, 1000.0, 0.217, 0.1, 2.0, 3.5, 4180.0, 0.6, 0.0, 0)

    # calculate wavenumber (cm^-1) in each layer:
    for ii in np.arange(0, II):
        Layer[ii].k = (2.0 * np.pi * Tx.f) / (100.0 * Layer[ii].c)

    if (verbose): print('\tk: %f' % Layer[0].k )


    ## Computational Grid ######################################################

    # axial location of equivalent source (cm)
    if (not np.isfinite(Tx.d)):
        z_start = 0.0
    else:
        z_start = Tx.d - np.sqrt( np.power(Tx.d,2) - np.power(Tx.a2,2) )
    if (verbose):
        print("\tz_start: ", z_start)


    # max axial distance (cm)
    Z = 7.0

    # max number of harmonics in simulation (use power of 2)
    kpower = 6
    KK = np.int( np.power(2, kpower) )

    # grid resolution in r-direction (points per wavelength)
    ppw_r   = 15

    # and z-direction
    ppw_z   = 10

    # radius of equivalent source (a_2')
    if ( np.isinf(Tx.d) ):
        a2p = Tx.a2
    else:
        a2p = Tx.a2 * (Tx.d - z_start) / np.sqrt( np.power(Tx.d,2) - np.power(Tx.a2,2) )

    # width (radius) of physical domain
    w = 3.0 * Tx.a2

    # wavelength (cm)
    lambda0 = 2.0 * np.pi / Layer[0].k

    # PML thickness
    th = 2.0 * lambda0

    # max radius of computational domain (cm)
    R = w + th

    # Gridpoints in r-dir
    kscale = 0.35
    JJ = np.ceil(ppw_r * np.power(KK, kscale) * R / lambda0)

    # Gridpoints in z-direction
    NN = np.ceil(ppw_z * (Z - z_start) / lambda0)
    if (verbose): print("\tNN: ", NN)

    # node vectors
    r = np.transpose( np.linspace(0.0, R, np.int(JJ)) )

    z = np.linspace(z_start, Z, np.int(NN))

    Grid = GridClass(Z, np.int(KK), R, np.int(JJ), np.int(NN), r, z)

    dr = Grid.r[1]

    dz = Grid.z[1] - Grid.z[0]


    ## Graphical output ########################################################

    # locations on z-axis where plots are produced
    z0 = z_start
    z1 = 5.0
    z_output = np.array([z0, z1])

    # code determines number of plot locations
    LL = np.size(z_output)

    # create instances of SpecOut class
    SpecOut = np.ndarray((LL,), dtype=np.object)
    for ll in np.arange(0, LL):
        SpecOut[ll] = SpecOutClass(1, 1, 1, 1, 1)

    # initialize index
    ll = 0


    ##################################
    ## Equivalent source #############
    ##################################


    # This source is a converging spherical wave at z=z_start bounded by a2p.
    # Note - it's dimensionless
    if (np.isinf(Tx.d)):
        if (verbose): print("Planar ")
        A   = np.ones((Grid.JJ,)) * (Grid.r < a2p)
        a1p = Tx.a1
    else:
        if (verbose): print("Curved ")
        A = Tx.d * np.exp(-1j*Layer[0].k * np.sqrt( np.power(Grid.r,2) + np.power(Tx.d-z_start,2))) / np.sqrt( np.power(Grid.r,2) + np.power(Tx.d-z_start,2) ) * (Grid.r < a2p)
        # if there's a hole in the Tx
        if (Tx.a1 != 0):
            if (verbose): print("Aperture ")
            a1p = Tx.a1 * (Tx.d - z_start) / np.sqrt( np.power(Tx.d,2) - np.power(Tx.a1,2) )
            A   = A * (Grid.r > a1p)

    # The user could specify a custom source here [any complex function A=A(r)].
    A = np.exp( -(1.0 / np.power(Tx.a2,2) + 1j * Layer[0].k / 2.0 / Tx.d ) * np.power(Grid.r,2) )

    if (debug): print("\t max A:", np.max(A), '\n')

    if (debug): print( "\t shape A: ", np.shape(A) )

    # Apply a low-pass filter to the source
    A = SourceFilterH.SourceFilterH(Grid.r, A, Layer[0].k)

    if (debug): print( "\t shape A: ", np.shape(A) )

    # Next scale the source by the appropriate pressure coefficient so that it has
    # the proper total acoustic power
    integral = 2.0 * np.pi * dr * np.trapz( np.power(np.abs(A), 2) * Grid.r)

    if (debug): print( "\t shape integral : ", np.shape(integral) )

    # convert units from cm^2 to m^2
    integral = 1e-4 * integral
    p0       = np.sqrt(2.0 * Layer[0].rho * Layer[0].c * Tx.P / integral)
    # dimensionalize the boundary condition (units of Pa)
    A        = p0*A

    if (debug): print( "\t shape A: ", np.shape(A) )

    ## Spatial averaging #######################################################

    # convert to radius in cm
    hr = 0.1 * hd / 2.0


    ## Calculate attenuation, dispersion #######################################

    # vector of frequencies
    v = Tx.f * np.arange(1,Grid.KK+1) / 1e6
    if (debug): print( "np.shape v: ", np.shape(v) )
    for ii in np.arange(0, II):
        # convert to Np/cm
        Layer[ii].alpha = Layer[ii].alpha / 8.686 / 100
        # as v is a vector of frequencies
        if (Layer[ii].eta == 1):
            # linear media
            Layer[ii].alpha = Layer[ii].alpha * v * (1.0 + 2.0*1j*np.log(v) / np.pi)
        else:
            # everything else
            Layer[ii].alpha = Layer[ii].alpha * (np.power(v, Layer[ii].eta) - 1j*np.tan(np.pi*Layer[ii].eta/2.0) * ( np.power(v,Layer[ii].eta) - v ) )

    # some reporting
    if (output):
        print('\n\tWavelength = %3.2f [mm]' % np.float(10.0*lambda0) )
        print('\tNode count')
        print('\t\tAxial %d' % Grid.NN)
        print('\t\tRadial %d' % Grid.JJ)
        print('\tGrid stepsize')
        print('\t\tdz = %3.2f [mm]' % (10.0*dz) )
        print('\t\tdr = %3.2f [mm]' % np.float(10.0*dr))

    ## dependent variable (pressure) matrices -
    # new pressure, i.e. all r values at n+1 th step along z-axis for each computed harmonic
    p = np.zeros((Grid.JJ, Grid.KK), dtype=np.complex)

    # old pressure, i.e. at n th step
    q = np.zeros((Grid.JJ, Grid.KK), dtype=np.complex)

    # apply boundary condition (linear source)
    q[:,0] = A

    # mesh spacing near PML
    dr_max = Grid.r[Grid.JJ-1] - Grid.r[Grid.JJ-2]

    # Index of radial limit where spatial averaging occurs
    JJ_ = Grid.JJ - np.ceil(hr / dr_max)

    # peak axial rarefactional pressure
    p_r = np.zeros((Grid.NN,))

    # peak axial compressional pressure
    p_c = np.zeros((Grid.NN,))

    # recorded data
    p5 = np.zeros((np.min([Grid.KK, minharmonics]), Grid.NN), dtype=np.complex)

    # get pressure values
    if (hd==0):
        p_r[0] = np.min( np.real( q[:,0] ) )
        p_c[0] = np.max( np.real( q[:,0] ) )
        p5[0,0] = np.max(np.abs(A))
    else:
        p_r[0], p_c[0], p5[:,0] = SynthAxScan.SynthAxScan(Grid.r, q, hr, JJ_, 2*Grid.KK)

    # intensity
    I = np.zeros((Grid.JJ, Grid.NN))

    # power density
    Q = np.zeros((Grid.JJ, Grid.NN))

    #-------------------------------------------------------------------------------

    phi = (2.0 / th) * (Grid.r - Grid.R + th) * (Grid.r > (Grid.R-th) )
    phi = phi + (1.0 - phi) * (Grid.r > (Grid.R-th/2.0))
    u   = np.exp(-1j * np.pi * phi / 4.0)
    if (debug): print("shape u: ", np.shape(u) )

    Du2 = np.zeros((Grid.JJ, ))
    Du2 = (-1j * np.pi / 2.0 / th) * u * (Grid.r > Grid.R)
    Du2 = Du2 * (Grid.r < (Grid.R+th/2.0))
    if (debug): print("shape Du2: ", np.shape(Du2) )

    temp = np.zeros((np.size(Grid.r),), dtype = np.complex)
    temp[1:] = u[1:] / Grid.r[1:]

    ur = diags( temp )

    u  = diags( u )
    Du = diags( Du2 )

    # Build transverse Laplacian operator w/PML:
    del A
    e  = np.ones( (Grid.JJ, ) )
    D1 = diags(np.array([-e, e]),        np.array([-1,1]),   np.array([Grid.JJ, Grid.JJ]) ) / 2.0 / dr
    D2 = diags(np.array([e, -2.0*e, e]), np.array([-1,0,1]), np.array([Grid.JJ, Grid.JJ]) ) / dr / dr
    A = u * ( (ur + Du) * D1 + u * D2)

    # zero flux BC at r=0
    A[0,1] = 2.0 * A[0,1]

    #-------------------------------------------------------------------------------

    # peripherals for nonlinear integrator -
    Ppos      = np.zeros((Grid.JJ, Grid.NN))
    Ppos[:,0] = np.abs(q[:,0])
    Pneg      = np.zeros((Grid.JJ, Grid.NN))
    Pneg[:,0] = -np.abs(q[:,0])

    p5[0,0]   = np.abs(q[0, 0])

    # waveform data vectors -
    w         = np.zeros((Grid.NN, 2*Grid.KK), dtype=np.complex)
    Y         = np.zeros((2*Grid.KK,), dtype=np.complex)

    # change in intensity
    I_td      = np.zeros((Grid.JJ, 2))

    # in seconds
    dt        = 1.0 / Tx.f / (2.0*Grid.KK - 1)

    # in us
    t         = 1e6 * np.linspace(0.0, 1.0/Tx.f, num=np.int(2*Grid.KK), dtype=np.float64)
    if (debug): print( "np.shape t: ", np.shape(t) )

    # more reporting
    if (output):
        print('\t\tdt = %3.2f [ns]\n' % (1e9*dt) )

    # find indices of first gridpoint in each Layer. do first, then loop through middle layers, then add another for ghost last layer
    Layer[0].index = 0
    if (II > 1):
        for ii in np.arange(1,II-1):
            Layer[ii].index = np.int( np.ceil((Layer[ii].z - z_start)/dz)) + np.int(1)
        # do last
        Layer[II-1].index = np.int(Grid.NN)

    if (verbose):
        for i in np.arange(0,II, dtype=np.int):
            print("Layer.index: ", Layer[i].index )

    if (debug):
        if (II > 1):
            print( "layer indices:", Layer[0].index, Layer[1].index )
        else:
            print( "single layer" )

    #-------------------------------------------------------------------------------

    pade = '1'

    # integration loop:
    for ii in np.arange(0, II, dtype=np.int):

        if (pade== '1'):
            op = [BuildPade11operators.BuildPade11operators(A, kk, dz, Layer[ii].k, Grid.JJ) for kk in np.arange(1,Grid.KK+1)]
            P1, P2 = zip(*op)
        if (pade== '2'):
            op = [BuildPade12operators.BuildPade12operators(A, kk, dz, Layer[ii].k, Grid.JJ) for kk in np.arange(1,Grid.KK+1)]
            P1, P2, P3 = zip(*op)

        mu     = ( Layer[ii].beta / 2 / Layer[ii].rho / np.power(Layer[ii].c,3) )* (0.01 * dz / dt)

        # cutoff for nonlinearity
        cutoff = Layer[ii].alpha[0] * Layer[ii].rho * np.power(Layer[ii].c,2) / Layer[ii].beta / Layer[ii].k

        # ensure last index is okay
        if (II==1):
            upper = np.int(NN-1)
        else:
            if (ii==II-1):
                upper = np.int(NN-1)
            else:
                upper = np.int(Layer[ii+1].index-1)

        for nn in np.arange(Layer[ii].index, upper, dtype=np.int):

            # integrate nonlinear term -
            p, w[nn+1,:], Ppos[:,nn+1], Pneg[:,nn+1], I_td[:,0] = TDNL.TDNL(q, w[nn+1,:], Y, Grid.KK, Grid.JJ, mu, cutoff, Ppos[:,nn], Pneg[:,nn], I_td[:,0] )

            # attenuation/dispersion term and diffraction term:
            for kk in np.arange(0, Grid.KK):
                p[:,kk] = p[:,kk] * np.exp(-Layer[ii].alpha[kk] * dz)
                # for Pade 12
                if (pade=='2'):
                    p[:,kk] = spsolve( P1[kk], spsolve( P2[kk],  P3[kk].dot(p[:,kk]) ) )
                # for Pade 11
                if (pade=='1'):
                    p[:,kk] = spsolve( P1[kk], P2[kk].dot(p[:,kk]) )

            # compute norm of solution
            Norm = np.linalg.norm( p[:,0] )

            # stop if something goes wrong
            if not np.isfinite(Norm):
                print( colored('\tNaN or Inf detected!', 'red'), 'Simulation stopped at z = %d cm.\n' % Grid.z[nn])

            # update data from n to n+1
            q = p

            # calculate intensity I and power density H
            for jj in np.arange(0, Grid.JJ):
                I[jj,nn+1] = np.sum( np.power( np.abs(p[jj,:]),2) ) / 2.0 / Layer[ii].rho / Layer[ii].c
                # check this!
                sum1 = np.sum( Layer[ii].fraction * np.real(Layer[ii].alpha) * np.power(np.abs(np.transpose(p[jj,:])),2) )
                sum2 = np.sum( I_td[jj,:] ) / (2.0 * dz * (Grid.KK-1) )
                #Q[jj,nn+1] = ( np.sum( Layer[ii].fraction * np.real(Layer[ii].alpha) * np.power(np.abs(np.transpose(p[jj,:])),2) ) + np.sum(I_td[jj,:]) / dz / (2.0*Grid.KK-1) ) / Layer[ii].rho / Layer[ii].c
                Q[jj,nn+1] = ( sum1 + sum2 ) / (Layer[ii].rho * Layer[ii].c)
                #print( sum1, sum2)

            # collect/process data:
            if (hd == 0):
                p_r[nn+1] = np.min( np.real(w[nn+1,:]) )
                p_c[nn+1] = np.max( np.real(w[nn+1,:]) )
                p5[:,nn+1] = np.transpose( p[0, 0:np.min([Grid.KK,minharmonics]) ] )
            else:
                p_r[nn+1], p_c[nn+1], p5[:,nn+1] = SynthAxScan.SynthAxScan(Grid.r, p, hr, JJ_, 2*Grid.KK)

            if (debug):
                print( nn, nn+1, p_r[nn+1], p_c[nn+1], np.min( np.real(w[nn+1,:]) ), np.max( np.real(w[nn+1,:]) ) )
                kk=np.int(0)
                print("****", np.shape(P1[kk]), np.shape(P2[kk]), np.shape(p), np.shape(P2[kk] * p[:,kk] ), np.max(P1[kk]), np.max(P2[kk]), "****"  )

            if (ll < LL):
                if (debug): print("here 0: \t", ll, LL, z_output[ll], nn, Grid.z[nn], Grid.z[nn+1], hd )
                # find special output locations
                if ( (Grid.z[nn+1] > z_output[ll]) & (Grid.z[nn] <= z_output[ll]) ):
                    if (debug): print("here 1: \t", ll, LL, z_output[ll], nn, Grid.z[nn], hd )
                    if ((hd == 0) or (np.abs(hd) < 10E-8)):
                        if (debug): print("here 2: \t", ll, LL, z_output[ll], nn, Grid.z[nn], hd )
                        SpecOut[ll].p_r = Pneg[:,nn+1]
                        SpecOut[ll].p_c = Ppos[:,nn+1]
                        SpecOut[ll].p5 = np.abs( p[:, 0:np.min([minharmonics,Grid.KK])] )
                        SpecOut[ll].w  = w[nn+1,:]
                    else:
                        if (debug): print("here 3: \t", ll, LL, z_output[ll], nn, Grid.z[nn], hd )
                        SpecOut[ll] = SynthRadScan.SynthRadScan(Grid.r, p, hr, JJ_, SpecOut[ll])

                    SpecOut[ll].I = I[:,nn+1]
                    if (debug): pprint( vars(SpecOut[ll]) )
                    ll = ll + 1


        # rescale pressure due to transmission at interface between layers ii and ii+1:
        if ( (ii < (II-1)) and (II > 1) ):
            q = 2.0*Layer[ii+1].rho * Layer[ii+1].c * q / (Layer[ii].rho * Layer[ii].c + Layer[ii+1].rho * Layer[ii+1].c)

    # integration loop ends
    tend  = timeit.default_timer()
    times = tend - tstart
    h     = np.floor(times / 3600)
    times = times - h * 3600
    m     = np.floor(times / 60)
    times = times - m * 60

    if (output):
        print('\t\t%02d:%02d:%04.1f' %(h, m, times) )
        print('\t\tTook %2.1f seconds.' %(tend-tstart) )

    #Layer = Layer[0:II]


    if (debug):
        print( np.shape( SpecOut[ll-1].w ), np.shape(t) )
        print( t )

    ############################
    ## Plot routine ############
    ############################

    HH = np.min( np.shape(p5) )
    MM = np.max( np.shape(SpecOut[0].p5) )

    if (debug):
        pprint( vars(SpecOut[0]) )
        print( np.shape( p5 ), np.shape(SpecOut[0].p5), np.shape( Grid.r ) )

    if (willPlot):

        # 3d plot showing axial and radial pressure amplitudes
        fig = plt.figure()
        ax = Axes3D(fig)
        r_ones  = np.ones((MM,) )
        z_zeros = np.zeros((np.size(Grid.z),))
        for jj in np.arange(0, HH, dtype=np.int):
            ax.plot3D(Grid.z, z_zeros, np.abs(p5[jj,:])/1e6, linewidth=2)
        for ll in np.arange(0, LL):
            plt.gca().set_prop_cycle(None)
            for jj in np.arange(0, HH, dtype=np.int):
                ax.plot3D(z_output[ll]*r_ones, Grid.r[0:np.size(SpecOut[ll].p5[:,jj])], SpecOut[ll].p5[:,jj]/1e6, linewidth=2)
        ax.set_xlabel('z (cm)')
        ax.set_ylabel('r (cm)')
        ax.set_zlabel('|p| (MPa)')
        plt.grid(True)

        # axial plots of amplitude of first 5 harmonics and intensity
        plt.figure()
        plt.subplot(2,1,1)
        for jj in np.arange(0, HH, dtype=np.int):
            plt.plot(Grid.z, np.abs(p5[jj,:])/1e6, linewidth=2)
        plt.ylabel('|p| (MPa)')
        plt.grid(True)
        plt.subplot(2,1,2)
        plt.plot(Grid.z, I[0,:]/1e4, linewidth=2)
        plt.xlabel('z (cm)')
        plt.ylabel('I (W/cm^2)')
        plt.grid(True)

        if (Grid.KK > 1):
            if (debug): print( np.shape(SpecOut[0].w), np.shape(t) )
            # build plot label
            V = []
            for ll in np.arange(0, LL):
                V.append('z= ' + str(z_output[ll]) + ' cm')
            # temporal waveforms at specified axial locations
            plt.figure()
            for ll in np.arange(0,LL):
                plt.plot(t, np.real( SpecOut[ll].w )/1e6, linewidth=2, label=V[ll])
            plt.xlim([t[0], t[np.size(t)-1]])
            #plt.legend(V)
            plt.grid(True)
            plt.xlabel('t (us)')
            plt.ylabel('p (MPa)')
            # axial plots of compressional and rarefactional pressure
            plt.figure()
            #if (hd == 0):
                #p_c = Ppos[0,:]
                #p_r = Pneg[0,:]
            plt.plot(Grid.z, p_c/1e6, linewidth=2)
            plt.plot(Grid.z, p_r/1e6, linewidth=2)
            plt.xlabel('z (cm)')
            plt.ylabel('p (MPa)')
            plt.grid(True)

        for ll in np.arange(0, LL):
            plt.figure()	# radial plots of amplitude of first 5 harmonics and intensity
            plt.subplot(2,1,1)	# at specified axial locations
            for jj in np.arange(0, HH, dtype=np.int):
                plt.plot(Grid.r[0:np.size(SpecOut[ll].p5[:,jj])], SpecOut[ll].p5[:,jj]/1e6, linewidth=2)
            plt.ylabel('|p| (MPa)')
            plt.title(V[ll])
            plt.grid(True)
            plt.subplot(2,1,2)
            plt.plot(Grid.r, SpecOut[ll].I/1e4, linewidth=2)
            plt.xlabel('r (cm)')
            plt.ylabel('I (W/cm^2)')
            plt.grid(True)

        # spatial distribution of field emphasizing low-amplitude variations
        plt.figure()
        r = np.hstack( (-Grid.r[Grid.JJ:1:-1], Grid.r) )
        I = np.vstack( (I[Grid.JJ:1:-1,:], I) )
        plt.pcolor(Grid.z, r, np.power(I,0.2) )
        for ll in np.arange(0, LL):
            plt.axvline(x=z_output[ll], color='red')
        plt.xlabel('z (cm)')
        plt.ylabel('r (cm)')
        plt.grid(True)

        plt.show()

    return [Grid, Layer, Q]
