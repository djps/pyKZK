import numpy as np

def SynthRadScan(r, p, b, JJ_, verbose=False, nharmonics=5):
    """
    r = radial node vector (cm)
    p = pressure matrix (radial x harmonic spectrum)
    b = hydrophone element radius (cm)

    returns SpecOut, a structure containing p_r and p_c, vectors of averaged
    peak rarefactional and compressional pressure and first (up to) 5
    averaged harmonic pressure amplitudes, all as a function of radius.
    Also contains the averaged waveform on axis.
    """

    class SpecOutClass(object):
        pass

    # JJ = number of radial nodes; KK = number of harmonics
    [_, KK] = np.shape(p)

    # mesh spacing near axis
    dr_min  = r[1]
  
    debug = False
    if (debug): 
        print(JJ_)
        print(KK)
        print( type(np.ceil(10*b/dr_min) ) )

    # matrix of spatially averaged pressure values
    p_h = np.zeros( (np.int(JJ_),np.int(KK)), dtype=np.complex )

    # number of points over which to spatially average
    NN = np.max( [np.int(4), np.int(np.ceil(10*b/dr_min))] )

    dr = b / (NN - 1)

    x = np.linspace(0, b, np.int(NN) )

    q = np.zeros((np.int(NN),KK), dtype=np.complex)

    U = np.zeros((np.int(2*KK),), dtype=np.complex)

    for kk in np.arange(0, KK):
        if debug: print( kk, np.shape(q[:,kk]), np.shape(x), np.shape(r), np.shape( p[:,kk] ), np.shape( np.interp(x, r, p[:,kk]) ) )
        q[:,kk]   = np.interp(x, r, p[:,kk])
        p_h[0,kk] = dr * np.trapz(q[:,kk] * x)


    for jj in np.arange(1,np.int(JJ_), dtype=np.int):
        if (r[jj] < b):
            # if element overlays central axis

            # setup for "inner circle"
            lowerlimit = 0
            upperlimit = b - r[jj]
            NN = np.ceil(10.0 * (upperlimit - lowerlimit) / dr_min)
            dr = (upperlimit - lowerlimit) / (NN - 1)
            x = np.linspace(lowerlimit, upperlimit, np.int(NN))
            q = np.zeros((np.int(NN),KK), dtype=np.complex)
            for kk in np.arange(0, KK, dtype=np.int):
                if (debug): print( kk, np.shape(x), np.shape(r), np.shape(p[:,kk]) )
                q[:,kk] = np.interp(x, r, p[:,kk])
                p_h[jj,kk] = dr * np.trapz( np.conj( np.transpose(q[:,kk]) ) * x )

            # setup for outer crescent
            lowerlimit = b - r[jj]
            upperlimit = r[jj] + b
            NN = np.ceil( 10*(upperlimit - lowerlimit) / dr_min)
            dr = (upperlimit - lowerlimit) / (NN - 1)
            x = np.linspace(lowerlimit, upperlimit, np.int(NN))
            q = np.zeros((np.int(NN),KK), dtype=np.complex)
            for kk in np.arange(0, KK, dtype=np.int):
                q[:,kk]    = np.interp(x, r, p[:,kk])
                W = weight(x, r[jj], b)
                p_h[jj,kk] = p_h[jj,kk] + dr * np.trapz( np.conj( np.transpose( q[:,kk] )) * W * x)

        else:
            lowerlimit = r[jj] - b
            upperlimit = r[jj] + b
            NN = np.ceil(10*(upperlimit-lowerlimit)/dr_min)
            dr = (upperlimit-lowerlimit)/(NN-1)
            if (debug): print( upperlimit, lowerlimit, NN, r[jj], b, dr_min)
            x = np.linspace(lowerlimit, upperlimit, np.int(NN))
            q = np.zeros((np.int(NN),KK), dtype=np.complex)
            for kk in np.arange(0, KK, dtype=np.int):
                if (debug): print( np.shape(x), np.shape(r), np.shape(p[:,0]), np.shape(q[:,kk]), np.shape(p_h[jj,kk]) )
                q[:,kk]    = np.interp(x, r, p[:,kk])
                W = weight(x, r[jj], b)
                p_h[jj,kk] = dr * np.trapz( np.conj( np.transpose(q[:,kk] )) * W * x)

    p_h = 2.0*p_h/b/b
    p5  = np.abs( p_h[:,0:np.min([nharmonics,KK])] )

    p_r = np.zeros((np.int(JJ_),))
    p_c = np.zeros((np.int(JJ_),))

    # determine peak compressional p_c and rarefactional p_r pressure
    if (KK == 1):
        # linear case - do nothing
        for jj in np.arange(0,JJ_,dtype=np.int):
            p_c[jj] = np.abs( p_h[jj,0] )
        p_r = -p_c
    else:
        # nonlinear case - transform to time domain
        # in each radial node jj
        for jj in np.arange(JJ_-1, 0, -1, dtype=np.int):
            if debug: print( JJ_, JJ_-1, jj, np.shape(U), np.shape( U[1:KK+1] ), np.shape( np.conj( p_h[jj,:] ) ), np.shape( U[2*KK-1:KK+1:-1] ), np.shape(p_h[jj, 0:KK-2]) )
            U[1:KK+1]         = np.conj( p_h[jj,:] )
            U[2*KK-1:KK+1:-1] = p_h[jj, 0:KK-2]
            # transform to time domain:
            U = KK * np.fft.ifft(U)
            p_r[jj] = np.min( np.real(U) )
            p_c[jj] = np.max( np.real(U) )
            
    SpecOut = SpecOutClass()

    SpecOut.w = U
    SpecOut.pr = p_r
    SpecOut.pc = p_c
    SpecOut.p5 = p5
    SpecOut.I = p_r # placeholder; intensity is assigned in WAKZK()

    return SpecOut



def weight(r, r0, b):
    import warnings

    #np.seterr(all='print')

    arg = (r*r + r0*r0 - b*b) / (2.0*r0*r)
    cond = np.abs(arg-1.0)
    v = np.argwhere(cond >= 0.0) 
    x = np.zeros(np.shape(arg))
    # print("v =", v)
    # print("size v:", np.shape(v))
    # if (np.size(v,0) > 1): 
    #     print("\t", v[1,:] )
    # print("size arg:", np.size(arg))

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        for i in np.arange(0, np.size(arg) ):
            if (np.size(v,0) != 0):
                if i in v[0,:]:
                    x[i] = 0.0
                    print("replaced i", i, "with zero.")
            else:
                x[i] = np.real( np.arccos( arg[i] )/np.pi )
                    # try:
                    #     x[i] = np.real( np.arccos( arg[i] )/np.pi )
                    # except Warning as e:
                    #     print('Houston, we have a warning:', e)
                    #     print(i, )
                    #     x[i] = 0.0
                    #     pass


    # for i in np.arange(0, np.size(arg) ):
    #     try:
    #         x[i] = np.real( np.arccos( arg[i] )/np.pi )
    #     except Warning:
    #         print(i)
    #         x[i] = 0.0


    # for i in np.arange(0, np.size(arg) ):
    #     if (np.size(v,0) != 0):
    #         if i in v[0,:]:
    #             x[i] = 0.0
    #         else:
    #             x[i] = np.real( np.arccos( arg[i] )/np.pi )
    
    #c = np.size( x[np.isnan(x)] )
    
    # print(a, c, np.transpose(v), arg, x, np.isnan(x).sum() )
    
    #x[np.isnan(x)] = 0.0
    
    return x
