""" converter.py
David Sinden 2012
This file converts the axisymmetric data into 3D Cartesian data. In this
case the axial interpolation is done by the skip variable and the radial
interpolation is done by simply changing the intervals at which data is
read.
"""

import numpy as np

import os, time, datetime

import peakdetect

from scipy.io import loadmat, savemat
from math import pi, log

# clear screen
os.system('cls' if os.name == 'nt' else 'clear')

# load canonical data set
loadmat('Computed_Data/thermal_5s_-11dbm_1.mat')

# define colours for figures
ICRgray = [98, 100, 102]/255       #{ICRgray}{cmyk}{0.04,0.02,0,0.6}
ICRgreen = [202, 222, 2]/255       #{ICRgreen}{cmyk}{0.09,0,0.99,0.13}
ICRred = [166, 25, 48]/255         #{ICRred}{cmyk}{0.00 0.85 0.71 0.35}
ICRpink = [237, 126, 166]/255      #{ICRpink}{cmyk}{0.00 0.47 0.30 0.07}
ICRorange = [250, 162, 0]/255      #{ICRorange}{cmyk}{0.00 0.35 1.00 0.02}
ICRyellow = [255, 82, 207]/255     #{ICRyellow}{cmyk}{0.00 0.68 0.19 0.00}
ICRolive = [78, 89, 7]/255         #{ICRolive}{cmyk}{0.24 0.13 0.93 0.60}
ICRdamson = [79, 3, 65]/255        #{ICRdamson}{cmyk}{0.21 0.97 0.35 0.61}
ICRbrightred = [255, 15, 56]/255   #{ICRbrightred}{cmyk}{0 0.94 0.78 0}
ICRlightgray = [159, 168, 170]/255 #{ICRlightgray}{cmyk}{0.16 0.11 0.10 0.26}
ICRblue = [0, 51, 41]/255          #{ICRblue}{cmyk}{1.00 0.00 0.20 0.80}

verbose = 1

# Rescale quantities according to non-dimensionalisation scheme
Z = d*Z
R = a*R
p5x = 1e-6*p0*p5x
p5r = 1e-6*p0*p5r
peak = 1e-6*p0*peak
trough = 1e-6*p0*trough
K,N = np.size(p5x)

pks,loc = peakdetect( np.abs(trough), delta=15 )

# minpeakdistance is somewhat arbitary, but it enables the secondary peak
# to be identified.
dummy,ii = np.sort(pks)
axial_secondary_value = trough[loc[ii[-2]]]
axial_secondary_location = z[loc[ii[-2]]]

radial = np.sum( np.transpose( p5r[0:-1,:] ) )
pks,loc = findpeaks( radial, delta=3)
dummy,ii = np.sort(pks)
radial_secondary_value = np.sum( p5r[loc[ii[-1]],:] )
radial_secondary_location = r[loc[ii[-1]]]

#-------------------------------------------------------------------------#
threshold = -1.05 * np.maximum( (radial_secondary_value, axial_secondary_value) )
#-------------------------------------------------------------------------#

# get intersections of peak negative pressure threshold value and peak
# negative pressure field

# axial
[x0, y0, iout, jout] = intersections(z, threshold*np.ones(np.shape(z),1), z, trough)
iout = np.around(iout)

# radial
[x1, y1, iout1, jout1] = intersections( r[0:np.around(np.shape(r)/2)], threshold*np.ones(np.around(np.shape(r)/2),1), r[0:np.around(np.shape(r)/2)], -np.sum( p5r[0:np.around(np.shape(r)/2),:], 2 ) )
iout1 = np.around(iout1)

# # axial : peak positive and negative pressures
# figure
# hold on
# set[gca,'FontSize',12]
# fill[ [z[iout[1]:iout[2]]; z[iout[2]:-1:iout[1]]; z[iout[1]] ],
#       [peak[iout[1]:iout[2]]; trough[iout[2]:-1:iout[1]]; peak[iout[1]] ],
#        ICRbrightred, 'EdgeColor','r' ]
# fill[ [z[iout[3]:iout[4]]; z[iout[4]:-1:iout[3]]; z[iout[3]] ], [peak[iout[3]:iout[4]]; trough[iout[4]:-1:iout[3]]; peak[iout[3]] ], ICRbrightred, 'EdgeColor','r' ]
# plot[ z[iout[1]], trough[iout[1]], 'o', 'MarkerEdgeColor', ICRblue, 'MarkerFaceColor', ICRblue, 'MarkerSize', 6]
# plot[ z[iout[2]], trough[iout[2]], 'o', 'MarkerEdgeColor', ICRblue, 'MarkerFaceColor', ICRblue, 'MarkerSize', 6]
# plot[ z[iout[3]], trough[iout[3]], 'o', 'MarkerEdgeColor', ICRblue, 'MarkerFaceColor', ICRblue, 'MarkerSize', 6]
# plot[ z[iout[4]], trough[iout[4]], 'o', 'MarkerEdgeColor', ICRblue, 'MarkerFaceColor', ICRblue, 'MarkerSize', 6]
# plot[ axial_secondary_location, axial_secondary_value, '*', 'MarkerEdgeColor', ICRolive, 'MarkerFaceColor', ICRblue, 'MarkerSize', 6]
# plot[z,peak,z,trough,'LineWidth',2]
# plot[z, threshold*ones[np.shape(z],1],'LineWidth',1, 'LineStyle', '--', 'Color', ICRred]
# ylim = get[gca,'YLim'];
# plot[ [z[iout[1]],z[iout[1]]], [ylim[1],ylim[2]], 'LineWidth', 1, 'LineStyle', '--', 'Color', ICRblue],
# plot[ [z[iout[2]],z[iout[2]]], [ylim[1],ylim[2]], 'LineWidth', 1, 'LineStyle', '--', 'Color', ICRblue],
# plot[ [z[iout[3]],z[iout[3]]], [ylim[1],ylim[2]], 'LineWidth', 1, 'LineStyle', '--', 'Color', ICRblue],
# plot[ [z[iout[4]],z[iout[4]]], [ylim[1],ylim[2]], 'LineWidth', 1, 'LineStyle', '--', 'Color', ICRblue],
# axis[[0,Z,ylim[1],ylim[2]]]
# xlabel['z [cm]','FontSize',14]
# ylabel['p [MPa]','FontSize',14]
# title['Peak Pressures', 'FontSize', 18]
# set[gca,'XMinorTick','on','YMinorTick','on']
# grid on
#
# # radial : Note this is the positive value
# figure
# hold on
# fill[ [r[1:iout1[1]]; r[iout1[1]:-1:1] ], [ sum[p5r[1:iout1[1],:],2]; zeros[iout1[1],1] ], ICRbrightred, 'EdgeColor','r' ]
# #fill[ [r[iout1[2]:iout1[3]]; r[iout1[3]:-1:iout1[2]] ], [ sum[p5r[iout1[2]:iout1[3],:],2]; zeros[np.shape(iout1[2]:iout1[3]],1] ], ICRbrightred, 'EdgeColor','r' ]
# #fill[ [r[iout1[4]:iout1[5]]; r[iout1[5]:-1:iout1[4]] ], [ sum[p5r[iout1[4]:iout1[5],:],2]; zeros[np.shape(iout1[4]:iout1[5]],1] ], ICRbrightred, 'EdgeColor','r' ]
# plot[ r[1:round[np.shape(r]/2]], sum[ p5r[1:round[np.shape(r]/2],:], 2], 'LineWidth',2]
# plot[ r[1:round[np.shape(r]/2]], abs[threshold]*ones[round[np.shape(r]/2],1], 'LineWidth',1, 'LineStyle', '--', 'Color', ICRred],
# #plot[ 0, sum[p5r[1,:]], 'o', 'MarkerEdgeColor', ICRblue, 'MarkerFaceColor', ICRblue,  'MarkerSize', 6]
# plot[ radial_secondary_location, radial_secondary_value, '*', 'MarkerEdgeColor', ICRolive, 'MarkerFaceColor', ICRblue, 'MarkerSize', 6]
# plot[ r[iout1[1]], abs[threshold], 'o', 'MarkerEdgeColor', ICRblue, 'MarkerFaceColor', ICRblue, 'MarkerSize', 6]
# #plot[ r[iout1[2]], abs[threshold], 'o', 'MarkerEdgeColor', ICRblue, 'MarkerFaceColor', ICRblue, 'MarkerSize', 6]
# #plot[ r[iout1[3]], abs[threshold], 'o', 'MarkerEdgeColor', ICRblue, 'MarkerFaceColor', ICRblue, 'MarkerSize', 6]
# #plot[ r[iout1[4]], abs[threshold], 'o', 'MarkerEdgeColor', ICRblue, 'MarkerFaceColor', ICRblue, 'MarkerSize', 6]
# #plot[ r[iout1[5]], abs[threshold], 'o', 'MarkerEdgeColor', ICRblue, 'MarkerFaceColor', ICRblue, 'MarkerSize', 6]
# ylim = get[gca,'YLim'];
# plot[ [r[iout1[1]],r[iout1[1]]], [ylim[1],ylim[2]], 'LineWidth', 1, 'LineStyle', '--', 'Color', ICRblue],
# #plot[ [r[iout1[2]],r[iout1[2]]], [ylim[1],ylim[2]], 'LineWidth', 1, 'LineStyle', '--', 'Color', ICRblue],
# #plot[ [r[iout1[3]],r[iout1[3]]], [ylim[1],ylim[2]], 'LineWidth', 1, 'LineStyle', '--', 'Color', ICRblue],
# #plot[ [r[iout1[4]],r[iout1[4]]], [ylim[1],ylim[2]], 'LineWidth', 1, 'LineStyle', '--', 'Color', ICRblue],
# #plot[ [r[iout1[5]],r[iout1[5]]], [ylim[1],ylim[2]], 'LineWidth', 1, 'LineStyle', '--', 'Color', ICRblue],
# set[gca,'FontSize',12]
# xlabel['r [cm]','FontSize',14]
# ylabel['p [MPa]','FontSize',14]
# title['Radial Pressure','FontSize',18]
# set[gca,'XMinorTick','on','YMinorTick','on']
# grid
# hold off

###########################################################################
# Dose
###########################################################################

# set length of radial variable
n1 = np.shape( r_bht[1:-1:2])

# set length of axial variable
m1 = np.shape(z_bht)

# determine factors of axial length
xfactors1 = factor(np.shape(z_bht)-1)
print(xfactors1)

# # set skip to be maximum of greatest common, or product of previous prime factors
# skip1 = max[ prod[xfactors1[1:np.shape(xfactors1]-1]], xfactors1[np.shape(xfactors1]] ];
skip1 = 25;

# allocate memory for Cartesian variables
shifted_system1 = np.zeros(2*n1-1, 2*n1-1, np.shape(0:m1:skip1) )

# scaled mesh mesh
[x_cart1, y_cart1] = np.meshgrid( r_bht[1:-1:2]*np.sin(np.pi/4.0) )

# set counter
kk = 0

# Increment along axial direction
for k in np.arange(1, skip1, m1):
    # move counter
    kk = kk+1
    # Function to interpolate
    V1 = Dmat[:,k]
    # at each axial value interpolate
    for i in np.arange(1,n1):
        for j in np.arange(1,n1):
		# quadrant 1
		f = scipy.interpolate.interp1d( r_bht[0:-1:2], V1[0:-1:2] )
		shifted_system1[n1-1+i, n1-1+j, kk] = f( sqrt[ x_cart1[1,i]**2 + y_cart1[j,1]**2 )
		# quadrant 2
        shifted_system1[n1-1+i, n1+1-j, kk] = shifted_system1[n1-1+i, n1-1+j, kk]
        # quadrant 3
        shifted_system1[n1+1-i, n1+1-j, kk] = shifted_system1[n1-1+i, n1-1+j, kk]
        # quadrant 4
        shifted_system1[n1+1-i, n1-1+j, kk] = shifted_system1[n1-1+i, n1-1+j, kk]

    if (verbose == 1)
        # output progress
        if (k == 1):
            print('\n')
        if (kk < 10):
            print('\tkk = %d \t\tk = %d\n', kk, k)
        else:
            print('\tkk = %i \tk = %i\n', kk, k)

        #     if[k==m1]
        #         fprintf['\n']


xn1,yn1,zn1 = np.size(shifted_system1)

Xgrid1,Ygrid1,Zgrid1 = np.meshgrid[0:xn1, 0:yn1, 0:zn1]

###########################################################################
# Peak Negative
###########################################################################

# set length of radial variable
n2 = np.shape( r[0:4:-1] )

# set length of axial variable
m2 = np.shape(z)

# determine factors of axial length
xfactors2 = factor[np.shape(z)-1];

# set skip to be maximum of greatest common, or product of previous prime factors
skip2 = np.maximum( prod[xfactors2[0:np.shape(xfactors2)-1]], xfactors2[np.shape(xfactors2)] )
skip2 = 50
skip2 = np.around( np.shape(z)/35 )

# allocate memory for Cartesian variables
shifted_system2 = np.zeros(2*n2-1, 2*n2-1, np.shape(0:skip2:m2) )

# scaled mesh mesh
[x_cart2, y_cart2] = np.meshgrid[ r[0:4:-1]*np.sin(pi/4) ];

# set counter
kk = 0
# Increment along axial direction
for k in np.arange(0,skip2,m2):
    # move counter
    kk = kk+1
    # Function to interpolate
    V2 = Pneg[:,k]*1e-6*p0;
    # at each axial value interpolate
    for i in np.arange(0,1,n2):
        for j in np.arange(0,1,n2):
            # quadrant 1
            shifted_system2[n2-1+i,n2-1+j,kk] = interp1[r[0:4:-1], V2[0:4:-1], sqrt[ x_cart2[0,i]**2 + y_cart2[j,0]**2 ] ];
            # quadrant 2
            shifted_system2[n2-1+i,n2+1-j,kk] = shifted_system2[n2-1+i,n2-1+j,kk]
            # quadrant 3
            shifted_system2[n2+1-i,n2+1-j,kk] = shifted_system2[n2-1+i,n2-1+j,kk]
            # quadrant 4
            shifted_system2[n2+1-i,n2-1+j,kk] = shifted_system2[n2-1+i,n2-1+j,kk]

    if (verbose==1):
        # output progress
        if (k == 1):
            print('\n')
        if (kk < 10):
            print('\tkk = %d \t\tk = %d\n',kk,k)
        else:
            print('\tkk = %d \tk = %d\n',kk,k)
        if (k == m2)
            print()'\n')

xn2,yn2,zn2 = np.size(shifted_system2)
Xgrid2,Ygrid2,Zgrid2 = np.meshgrid[1:xn2,1:yn2,1:zn2]

###########################################################################
# Peak Positive
###########################################################################

# allocate memory for Cartesian variables
shifted_system3 = np.zeros(2*n2-1, 2*n2-1, np.shape(0:skip2:m2) )

# scaled mesh mesh
[x_cart2, y_cart2] = np.meshgrid[r[0:4:-1]*np.sin(pi/4.0)];

# set counter
kk = 0
# Increment along axial direction
for k in np.arange(0, skip2, m2):
    # move counter
    kk = kk+1
    # Function to interpolate
    V3 = Ppos[:,k]*1e-6*p0;
    # at each axial value interpolate
    for i in np.arange(0,1,n2):
        for j in np.aranmge(0,1,n2):
            # quadrant 1
            shifted_system3[n2-1+i,n2-1+j,kk] = interp1[ r[0:4:-1], V3[0:4:-1], np.sqrt( x_cart2[0,i]**2 + y_cart2[j,0]**2 ) ];
            # quadrant 2
            shifted_system3[n2-1+i,n2+1-j,kk] = shifted_system3[n2-1+i,n2-1+j,kk];
            # quadrant 3
            shifted_system3[n2+1-i,n2+1-j,kk] = shifted_system3[n2-1+i,n2-1+j,kk];
            # quadrant 4
            shifted_system3[n2+1-i,n2-1+j,kk] = shifted_system3[n2-1+i,n2-1+j,kk];

    if (verbose==1):
        # output progress
        if (k==1):
            fprintf['\n']

        if (kk<10):
            fprintf('\tkk = #i \t\tk = #i\n',kk,k)
        else
            fprintf('\tkk = #i \tk = #i\n',kk,k)
        if (k==m2):
            fprintf('\n')


figure
#-------------------------------------------------------------------------#
ndata  = abs[ permute[shifted_system2,[1,3,2]] ];
nXgrid = permute[Xgrid2,[1,3,2]]/size[Xgrid2,2];
nYgrid = permute[Ygrid2,[1,3,2]]/size[Ygrid2,1];
nZgrid = permute[Zgrid2,[1,3,2]]/size[Zgrid2,3];
[num idx] = max[ndata[:]];
[xx yx zx] = ind2sub[size[ndata],idx];
xslice = nXgrid[1,1,xx];
yslice = nYgrid[yx,1,1];
zslice = nZgrid[1,yx,1];
#h=slice[nZgrid, nYgrid, nXgrid, ndata, zslice, xslice, xslice];
#set[h,'FaceColor','interp','EdgeColor','none', 'DiffuseStrength',.8]
#-------------------------------------------------------------------------#
set[gca,'FontSize',12]
grid on

alpha[0.25];
[faces1,verts1,colors1] = isosurface[permute[Zgrid1,[1,3,2]]/size[Zgrid1,3],permute[Xgrid1,[1,3,2]]/size[Xgrid1,1],permute[Ygrid1,[1,3,2]]/size[Ygrid1,2],...
    permute[shifted_system1,[1,3,2]], 240, permute[Zgrid1,[1,3,2]]/size[Zgrid1,3] ];
patch['Vertices', verts1, 'Faces', faces1, ...
    'FaceVertexCData', colors1, ...
    'FaceColor', 'red', ...
    'edgecolor', 'none'];
alpha[0.5];
[faces2,verts2,colors2] = isosurface[permute[Zgrid2,[1,3,2]]/size[Zgrid2,3],permute[Xgrid2,[1,3,2]]/size[Xgrid2,1],permute[Ygrid2,[1,3,2]]/size[Ygrid2,2],...
    permute[shifted_system2,[1,3,2]], threshold, permute[Zgrid2,[1,3,2]]/size[Zgrid2,3] ];
p2 = patch['Vertices', verts2, 'Faces', faces2, ...
    'FaceVertexCData', colors2, ...
    'FaceColor', [0.5 0.5 0.5], ...
    'edgecolor', 'none'];
#colormap bone
# colormap autumn
lighting gouraud;
camlight;
camlight[-80,-10];
#set[p1,'FaceColor','interp', 'FaceAlpha','interp'];
#set[p2,'FaceColor','blue', 'FaceAlpha','interp', 'EdgeColor', 'none']# 'EdgeColor', 'interp', 'EdgeAlpha','interp'];#alpha['color'];
#alphamap['rampdown'];
view[3];
daspect[[1 1 1]];
axis tight
xlabel['z','FontSize',14]
ylabel['x','FontSize',14]
zlabel['y','FontSize',14]
set[gca,'XMinorTick','on','YMinorTick','on', 'ZMinorTick', 'On']
#hleg1 = leg-1[[p2, p1], '-1MPa','C_{240}', 'Location','NorthEastOutside'];
#set[hleg1,'FontSize',14];

figure
set[gca,'FontSize',12]
grid on
alpha[0.25];
colormap bone
[faces1,verts1,colors1] = isosurface[permute[Zgrid1,[1,3,2]]/size[Zgrid1,3],permute[Xgrid1,[1,3,2]]/size[Xgrid1,1],permute[Ygrid1,[1,3,2]]/size[Ygrid1,2],...
    permute[shifted_system1,[1,3,2]],240, permute[Zgrid1,[1,3,2]]/size[Zgrid1,3] ];
p1 = patch['Vertices', verts1, 'Faces', faces1, ...
    'FaceVertexCData', colors1, ...
    'FaceColor','red', ...
    'edgecolor', 'none'];
alpha[0.5];
[faces3,verts3,colors3] = isosurface[ permute[Zgrid2,[1,3,2]]/size[Zgrid2,3], permute[Xgrid2,[1,3,2]]/size[Xgrid2,1], permute[Ygrid2,[1,3,2]]/size[Ygrid2,2],...
    permute[shifted_system3,[1,3,2]], max[max[max[shifted_system3]]]/2, permute[Zgrid2,[1,3,2]]/size[Zgrid2,3] ];
p3 = patch['Vertices', verts3, 'Faces', faces3, ...
    'FaceVertexCData', colors3, ...
    'FaceColor','interp', ...
    'edgecolor', 'none'];
lighting phong;
camlight;
camlight[-80,-10];
view[3];
daspect[[1 1 1]];
axis tight
xlabel['z','FontSize',14]
ylabel['x','FontSize',14]
zlabel['y','FontSize',14]
set[gca,'XMinorTick','on','YMinorTick','on', 'ZMinorTick', 'On']
# hleg1 = leg-1[[p2, p1], 'fwhm','C_{240}', 'Location','NorthEastOutside'];
# set[hleg1,'FontSize',14];

max[verts1[:,1]]
max[verts3[:,1]]
min[verts1[:,1]]
min[verts3[:,1]]

#--------------------------------------------------------------------------

# figure
# [C,h] = contour3[shifted_system1, 240*ones[[100,1]]];

# ndata  = abs[ permute[shifted_system2,[1,3,2]] ];
# nXgrid = permute[Xgrid2,[1,3,2]];#/size[Xgrid2,3];
# nYgrid = permute[Ygrid2,[1,3,2]];#/size[Ygrid2,1];
# nZgrid = permute[Zgrid2,[1,3,2]];#/size[Zgrid2,2];
# [num idx] = max[ndata[:]];
# [xx yx zx] = ind2sub[size[ndata],idx];
# xslice = nXgrid[1,1,xx];
# yslice = nYgrid[yx,1,1];
# zslice = nZgrid[1,1,zx];
# #h = slice[nZgrid, nYgrid, nXgrid, ndata, yx, xx, zx];
# h=slice[nZgrid, nYgrid, nXgrid, ndata, 24, 186, 186];
# set[h,'FaceColor','interp',...
# 	'EdgeColor','none',...
# 	'DiffuseStrength',.8]


# ndata  = abs[ permute[shifted_system2,[1,3,2]] ];
# nXgrid = permute[Xgrid2,[1,3,2]]/size[Xgrid2,2];
# nYgrid = permute[Ygrid2,[1,3,2]]/size[Ygrid2,1];
# nZgrid = permute[Zgrid2,[1,3,2]]/size[Zgrid2,3];
# [num idx] = max[ndata[:]];
# [xx yx zx] = ind2sub[size[ndata],idx];
# xslice = nXgrid[1,1,xx];
# yslice = nYgrid[yx,1,1];
# zslice = nZgrid[1,yx,1];
# h=slice[nZgrid, nYgrid, nXgrid, ndata, zslice, xslice, xslice];
# set[h,'FaceColor','interp', 'EdgeColor','none', 'DiffuseStrength', 0.8]
#
# Sx = 24;
# Sy = 93;
# Sz = 93;
# cvals = 240;
# figure
# contourslice[permute[shifted_system1,[1,3,2]], Sx, Sy, Sz, cvals];

#--------------------------------------------------------------------------
# vectors of patches for DOSE
d13= [ ...
    [verts1[faces1[:,1],0]-verts1[faces1[:,2],0]], ...
    [verts1[faces1[:,1],1]-verts1[faces1[:,2],1]], ...
    [verts1[faces1[:,1],2]-verts1[faces1[:,2],2]] ];
d12= [ ...
    [verts1[faces1[:,0],0]-verts1[faces1[:,1],0]], ...
    [verts1[faces1[:,0],1]-verts1[faces1[:,1],1]], ...
    [verts1[faces1[:,0],2]-verts1[faces1[:,1],2]] ];
# cross-product [vectorized]
cr = np.cross(d13, d12)
# Area of each triangle
area1 = 0.5*np.sqrt( cr[0,:]**2 + cr[1,:]**2 + cr[2,:]**2 )
# Total area
totalArea1 = np.sum(area1)
# norm of cross product
crNorm1 = np.sqrt(cr[0,:]**2 + cr[1,:]**2 + cr[2,:]**2)
# centroid
zMean1 = [verts1[faces1[:,0],2] + verts1[faces1[:,1],2] + verts1[faces1[:,2],2]] / 3.0
# z component of normal for each triangle
nz1 = -cr[2,:]./crNorm1
# contribution of each triangle
volume1 = np.abs( (area1*zMean1)*nz1 )
# divergence theorem
totalVolume1 = np.sum(volume1)
# display volume to screen
print('\n\tTotal volume of dose %8.5f\n', totalVolume1)
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# vectors of patches for cavitating region
d23= [ ...
    [verts2[faces2[:,2],1]-verts2[faces2[:,3],1]], ...
    [verts2[faces2[:,2],2]-verts2[faces2[:,3],2]], ...
    [verts2[faces2[:,2],3]-verts2[faces2[:,3],3]] ];
d22= [ ...
    [verts2[faces2[:,1],1]-verts2[faces2[:,2],1]], ...
    [verts2[faces2[:,1],2]-verts2[faces2[:,2],2]], ...
    [verts2[faces2[:,1],3]-verts2[faces2[:,2],3]] ];
# cross-product [vectorized]
cr2 = cross[d23, d22,1];
# Area of each triangle
area2 = 0.5*sqrt[cr2[1,:].**2+cr2[2,:].**2+cr2[3,:].**2];
# Total area
totalArea2 = sum[area2];
# norm of cross product
crNorm2 = sqrt[cr2[1,:].**2+cr2[2,:].**2+cr2[3,:].**2];
# centroid
zMean2 = [verts2[faces2[:,1],3]+verts2[faces2[:,2],3]+verts2[faces2[:,3],3]]/3;
# z component of normal for each triangle
nz2 = -cr2[3,:]./crNorm2;
# contribution of each triangle
volume2 = abs[ [area2.*zMean2'].*nz2 ];
# divergence theorem
totalVolume2 = sum[volume2];
# display volume to screen
fprintf['\n\tTotal volume of cavitating region #8.5f\n',totalVolume2];
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# vectors of patches for FWHM
d33= [ ...
    [verts3[faces3[:,2],1]-verts3[faces3[:,3],1]], ...
    [verts3[faces3[:,2],2]-verts3[faces3[:,3],2]], ...
    [verts3[faces3[:,2],3]-verts3[faces3[:,3],3]] ];
d32= [ ...
    [verts3[faces3[:,1],1]-verts3[faces3[:,2],1]], ...
    [verts3[faces3[:,1],2]-verts3[faces3[:,2],2]], ...
    [verts3[faces3[:,1],3]-verts3[faces3[:,2],3]] ];
# cross-product [vectorized]
cr3 = cross[d33',d32',1];
# Area of each triangle
area3 = 0.5*sqrt[cr3[1,:].**2+cr3[2,:].**2+cr3[3,:].**2];
# Total area
totalArea3 = sum[area3];
# norm of cross product
crNorm3 = sqrt[cr3[1,:].**2+cr3[2,:].**2+cr3[3,:].**2];
# centroid
zMean3 = [verts3[faces3[:,1],3]+verts3[faces3[:,2],3]+verts3[faces3[:,3],3]]/3;
# z component of normal for each triangle
nz3 = -cr3[3,:]./crNorm3;
# contribution of each triangle
volume3 = abs[ [area3.*zMean3'].*nz3 ];
# divergence theorem
totalVolume3 = sum[volume3];
# display volume to screen
fprintf['\n\tTotal volume of FWHM #8.5f\n',totalVolume3];
#--------------------------------------------------------------------------
