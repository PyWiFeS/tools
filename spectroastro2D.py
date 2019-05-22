"""
From the 26 March meeting, the plan was:
1) Fix 2D separation and overall R flux ratio. Find best fit PSF.

Issues... the best fit PSF can't just be a Gaussian. It is naturally the convolution of
multiple functional forms, i.e. something that is positive everywhere. On a quick search, 
I can't find any obvious parameterisations. Options...

a: Just use the interpolated PSF with a correction for the companion. Problem: we don't
know how to correct for the companion, so will have to do this iteratively.

b: Use a "distortion map". 

c: Use a functional form that can be negative and don't worry about details. 

2) Extract spectra of A and B components. This is best done with a *good seeing* night and doesn't have to 
be done for every data set. Save these spectra.

3) Fix B spectrum, and using the PSFs from step (1) extract the 2D positions of the A and
B components.

Star: GaiaDR2 6110141563309613184
...
is: 
2.343 arcsec North
0.472 arcsec East

It is 3.726 mags fainter in Rp.
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import glob
import scipy.optimize as op
import scipy.signal as sig
import time
import multiprocessing
import pdb
plt.ion()

#Settings
multiprocess=False #Setting this for a macbook changes total time from ~9 to ~5 seconds. Only a moderte help!
MIN_PEAK=20
NPSF_PARAMS = 5
WAVE = np.arange(6400.0,7000.0,0.25)
ddir = '/Users/mireland/data/pds70/190225/' #!!! This comes from 
#ddir = '/Users/mireland/data/pds70/190225/' #From Marusa's reduction.
fns = np.sort(glob.glob(ddir + '*p11.fits'))
xscale = 1.0 #arcsec/pix
yscale = 0.5 #arcsec/pix

#---------------------------------
#Local function declarations

def PSF(p,x,y,companion_params=None):
    """A simple 2D PSF based on a Gaussian.
    
    Parameters
    ----------
    p: numpy array
        Parameters for the PSF.
        p[0]: x coordinate offset
        p[1]: x coordinate width
        p[2]: y coordinate offset
        p[3]: y coordinate width
        p[4]: Total flux
        p[5]: 2nd order symmetric term
        
    x: x coordinate in arcsec.
    y: y coordinate in arcsec.
    """
    xp = (x-p[0])/p[1]
    yp = (y-p[2])/p[3]
    if companion_params != None:
        xp_comp = (x-p[0]-companion_params[1])/p[1]
        yp_comp = (y-p[2]-companion_params[2])/p[3]
        return p[4]*(np.exp(-(xp**2 + yp**2)/2.0) + companion_params[0]*np.exp(-(xp_comp**2 + yp_comp**2)/2.0))
    else:
        return p[4]*np.exp(-(xp**2 + yp**2)/2.0)

def PSF_resid(p,x,y,data, gain=1.0, rnoise=3.0):
    "Residuals for fitting to a 1D Gaussian"
    return ((PSF(p,x,y) - data)/10.).flatten() #np.sqrt(np.maximum(y,0) + rnoise**2)

def lsq_PSF( args ):
    """
    Fit a Gaussian to data y(x)
    
    Parameters
    ----------
    args: tuple
        guess_p, xfit, yfit
    
    Notes
    -----
    nline: int
        index of this line
    guess_center: float
        initial guess position
    """
    fit = op.least_squares(PSF_resid, args[0], method='lm', \
            xtol=1e-04, ftol=1e-4, f_scale=[3.,1.,1.], args=(args[1], args[2], args[3]))
    #Check for unphysical solutions and set c_inv to zero for those solutions...
    c_inv = fit.jac.T.dot(fit.jac)
    return fit.x, c_inv

#---------------------------------
#Main "Script" code
pas = []
mjds = []
fits = []
sigs = []
yx_peak = np.zeros( (len(WAVE), 2), dtype=np.int)
peak_vals = np.zeros( len(WAVE) )

dds = []
#Loop through files and make a 2D fit.
for f in fns[-3:]:
    ff = pyfits.open(f)
    pas.append(ff[0].header['TELPAN'])
    mjds.append(ff[0].header['MJD-OBS'])
    dd = ff[0].data[:,8:-8,13:-2]
    dds += [dd]
    
    
    #Subtract off local sky contribution. Could be more sophisticated!
    meds = np.median(dd.reshape(dd.shape[0], dd.shape[1]*dd.shape[2]), axis=1).reshape(dd.shape[0],1,1)
    dd -= meds
    
    #Find the maxima in every column.
    for i in range(len(WAVE)):
        yx_peak[i] = np.unravel_index(np.argmax(dd[i]), dd[i].shape)
        peak_vals[i] = dd[i, yx_peak[i][0], yx_peak[i][1]]
    
    #Create the x and y arrays
    xs, ys = np.meshgrid(np.arange(dd.shape[2])*xscale, np.arange(dd.shape[1])*yscale)
    
    #Now fit to every wavelength
    for i in range(len(WAVE)):
        fit, sig = lsq_PSF( ([yx_peak[i,1]*xscale,1,yx_peak[i,0]*yscale,1,peak_vals[i]], xs, ys, dd[i]) )
        fits += [fit]
        sigs += [sig]

fits = np.array(fits)
fits = fits.reshape( (len(fns), len(WAVE), NPSF_PARAMS) )

good = np.where(np.median(fits[:,:,4], axis=1) > 100)[0]

#Now find an average offset as a function of wavelength.
NE_offset = np.zeros( (len(WAVE),2) )
for i in good:
    NE_offset[:,0] += np.cos(np.radians(pas[i]))*fits[i,:,2] + np.sin(np.radians(pas[i]))*fits[i,:,0]
    NE_offset[:,1] += np.cos(np.radians(pas[i]))*fits[i,:,0] - np.sin(np.radians(pas[i]))*fits[i,:,2]
NE_offset /= len(fns)
    

