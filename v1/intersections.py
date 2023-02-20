import scipy.interpolate as interpolate
import scipy.optimize as optimize
import numpy as np

"""
http://stackoverflow.com/questions/8094374/python-matplotlib-find-intersection-of-lineplots
"""

def intersections(x1,y1,x2,y2):

	iverbose = 0

	p1 = interpolate.PiecewisePolynomial(x1,y1[:,np.newaxis])
	p2 = interpolate.PiecewisePolynomial(x2,y2[:,np.newaxis])

	xs = np.r_[x1,x2]
	xs.sort()
	x_min = xs.min()
	x_max = xs.max()
	x_mid = xs[:-1] + np.diff(xs)/2.0
	roots = set()
	i = 0
	for val in x_mid:
		pdiff = lambda x: p1(x) - p2(x)
		i = i+1
		root, infodict, ier, mesg = optimize.fsolve( pdiff, val, full_output=True )
		if (iverbose == 1):
			print( i )
    		# ier==1 indicates a root has been found
			if ((ier == 1) and (x_min < root < x_max)):
				roots.add( root[0] )
	
	roots = list(roots)

	if (iverbose == 1):
		print ( np.column_stack( (roots, p1(roots), p2(roots) ) ) )
	
	return roots, p1(roots), p2(roots)
