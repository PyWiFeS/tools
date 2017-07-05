"""Here we create some fake data using templates, and try to fit to this data 
using process_stellar to extract the radial velocities using TODCOR"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import process_stellar
plt.ion()

dir = 'RV_templates/'
infiles = [dir + '9000g40p00k2v50.txt', dir + '5250g35p00k2v50.txt']
rvs = [50, -150]
#rvs = [-50, 150]
#rvs = [20, -50]
#rvs = [-20, 50]
fluxes = [1, .3]
spectrograph_R = 7000.
#spectrograph_R = 3000.
wavelims = [3900,5000]
snr = 200.0
#-------------------
dwave = 0.5*(wavelims[0] + wavelims[1])/spectrograph_R/2 #2 pixel sampling
nwave = int((wavelims[1]-wavelims[0])/dwave)
wave_pixelated = np.linspace(wavelims[0], wavelims[1], nwave) 

#Zero the spectrum
spect = np.zeros(nwave)

#Add in each template by interpolating onto the new wavelength grid.
for fname, rv, flux in zip(infiles, rvs, fluxes):
    wave_spect = np.loadtxt(fname)
    spect += flux*np.interp(wave_pixelated, wave_spect[:,0]*(1+rv/3e5), wave_spect[:,1]/np.median(wave_spect[:,1]))
    #import pdb; pdb.set_trace()
    
#Convolve to the resolution of the spectrograph (2 pixels)
g = 0.5**np.arange(-2,3)**2
spect = np.convolve(spect, g, mode='same')
spect += np.median(spect)*np.random.normal(size=len(spect))/snr

sig = np.ones_like(spect)*np.median(spect)/snr
results = \
    process_stellar.calc_rv_todcor(spect,wave_pixelated, sig, infiles, plotit=True, smooth_distance=201, window_divisor=20)
    
print(results)
print("Computed delta RV: {0:6.2f} +/- {1:6.2f}".format(results[0]-results[2], np.sqrt(results[1]**2 + results[3]**2)))
print("Actual delta RV: {0:6.2f}".format(rvs[0]-rvs[1]))
print("Computed mean RV: {0:6.2f} +/- {1:6.2f}".format((results[0]+results[2])*.5, np.sqrt(results[1]**2 + results[3]**2)/2))
print("Actual mean RV: {0:6.2f}".format((rvs[0]+rvs[1])*.5))