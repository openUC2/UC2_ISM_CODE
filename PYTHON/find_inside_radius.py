#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 22:30:34 2019

@author: bene
"""

from scipy import spatial
import numpy as np



my_seak_radius = 15
my_peaks = np.uint16(np.random.random((100,2))*100)

my_peak = [6, 30]

my_peaksshift = (my_peaks - my_peak)
peakds_in_radius = my_peaksshift[(my_seak_radius-np.sqrt(my_peaksshift[:,0]**2+my_peaksshift[:,1]**2))>0,:]


# draw circle
theta = np.linspace(0, 2*np.pi, 100)
r = np.sqrt(1.0)
x1 = r*np.cos(theta)*my_seak_radius
x2 = r*np.sin(theta)*my_seak_radius


plt.scatter(peakds_in_radius[:,0], peakds_in_radius[:,1])
plt.scatter(x1,x2)
plt.draw()

# find peak closest to my right
myindex = np.argmin(np.sqrt((peakds_in_radius[:,0]**2+peakds_in_radius[:,1]**2)))
myindex

A[index]