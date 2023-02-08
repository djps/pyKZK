import numpy as np
import warnings
from numpy.fft import fft, ifft

from numba import jit

def TDNL(u, U, X, KK, JJ, c, cutoff, Ppos, Pneg, I_td, verbose=False):
    """
    converts spectrum to one cycle of the time-domain waveform and integrates the invicid Burger's equation using upwind/downwind method with periodic boundary conditions. TDNL stands for Time Domain NonLinear.
    """

    # set peak and trough values to zero; in case they are not assigned later
    if (KK==1):
        # linear case - do nothing
        for jj in np.arange(0, JJ):
            Ppos[jj] = np.abs( u[jj,0] )
        Pneg = -Ppos

    else:
        # nonlinear case - enter loop
        for jj in np.arange(JJ-1, 0):
        #for jj in np.linspace(JJ-1, 0, num=JJ, dtype=np.int):

            # execute nonlinear step only if amplitude is above cutoff
            # row jj=1 is always computed so plots look nice
            I_td[jj] = 0.0

            if (np.abs(u[jj,0]) < cutoff/20.0):
                # if pressure is too low, skip nonlinear step
                pass
            else:

                if (verbose): print( np.shape(U), KK, 2*KK, np.shape(u[0,0:KK-1]), np.shape(U[KK:2*KK-1]), np.shape(X[0,:]) )
                if (verbose): print( "\t jj: ", jj, "\t0. U[0]: ", type(U[0]) )
                # convert from sin & cos representation to complex exponential
                U[0] = complex(0.0,0.0)
                U[1:KK+1] = np.conj( u[jj,:] )
                U[2*KK-1 : KK : -1] = u[jj, 0:KK-1]

                # transform to time domain:
                #U = complex( KK * np.real( np.fft.ifft(U) ), np.zeros((np.size(U),)))
                U = KK * np.real( np.fft.ifft(U) ) + 0j
                if (verbose): print( "\t jj: ", jj, "\t1. U[0]: ", type(U[0]), "\t X[0]: ", type(X[0]) )
                I_td[jj] = np.trapz( np.squeeze( np.abs(U)**2) )

                # determine how many steps necessary for CFL<1 (CFL<0.9 to be safe).
                PP = np.ceil(2.0 * c * np.max(np.abs(U)) / 0.9 )

                # Nonlinear integration (upwind/downwind) algorithm. Note that p runs from 0 to P-1,
                # which is P steps in total, as calculated by the CFL condition
                for _ in np.arange( 0, PP ):

                    # for each frequency component
                    for kk in np.arange( 0, 2*KK, dtype=np.int ):

                        if ( np.real( U[kk-1] ) < 0.0 ):

                            if (kk == 0):
                                X[kk] = U[kk] + c*( U[0]*U[0] - U[2*KK-1]*U[2*KK-1] ) / PP / 2.0
                            else:
                                X[kk] = U[kk] + c*( U[kk]*U[kk] - U[kk-1]*U[kk-1] ) / PP / 2.0

                        else:

                            if ( kk == (2*KK-1) ):
                                X[kk] = U[kk] + c*( U[0]*U[0] - U[kk]*U[kk] ) / PP / 2.0
                            else:
                                X[kk] = U[kk] + c*( U[kk+1]*U[kk+1] - U[kk]*U[kk] ) / PP / 2.0

                    # update output argument U
                    U = X

                # account for nonlinear losses:
                I_td[jj] = I_td[jj] - np.trapz( np.squeeze( np.abs(X) )**2 )

                # store maximum and minimum
                Ppos[jj] = np.max( np.abs(X) )
                Pneg[jj] = np.min( np.abs(X) )

                # transform back to frequency domain:
                X = np.fft.fft(X) / KK

                if (verbose): print( "\t jj: ", jj, "\t2. X[0]: ", type(X[0]), "\t U[0]: ", type(U[0]) )

                # convert back to sin & cos representation:
                u[jj,:] = np.conj( X[1:KK+1] )


    return u,U,Ppos,Pneg,I_td
