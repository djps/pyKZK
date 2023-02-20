# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 12:05:00 2016

@author: djps

"""

import unittest

from axisymmetricKZK import axisymmetricKZK 
from axisymmetricBHT import axisymmetricBHT 

class SimpleTestCase(unittest.TestCase):

    def setUp(self):
        """Call before every test case."""
        drive = 3.0 
        efficiency = 0.74
        self.KZK = axisymmetricKZK(drive, efficiency) 
        
        #self.file = open( "blah", "r" )

    def tearDown(self):
        """Call after every test case."""
        self.file.close()

    def test_runner(self):
        """Test case A. note that all test method names must begin with 'test.'"""
        
        # compute acoustic field
        z,r,H,I,Ppos,Pneg,Ix,Ir,p0,p5r,p5x,\
         peak,trough,rho1,rho2,c1,c2,R,d,Z,M,\
         a,m_t,f,Xpeak,z_peak,K2,z_ = axisymmetricKZK.axisymmetricKZK(drive,efficiency);

        axisymmetricBHT.axisymmetricBHT(H, z, r, drive, efficiency)
        
        assert foo.bar() == 543


