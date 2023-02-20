def timing(p1,p2,t_start,z,d,n):
  
  """"
  
  Time keeping routine, formats output to screen.
  
  """
  
  from numpy import floor
  from time import time
  
  if ( p2 >= p1 ):
    times = time() - t_start 
    h    = floor(times/3600) 
    times = times - h*3600
    m    = floor(times/60)
    times = times - m*60
    p1   = p2
    
    print '\t%3.1f\t\t%02d:%02d:%04.1f\t\t%03d' %(z*d, h, m, times, n)
    
    del times, h, m

    return p1
