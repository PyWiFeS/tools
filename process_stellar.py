"""After analysis with WiFeS, this suite of routines can extract a star optimally
and calculate its radial velocity.

WARNING - this code is extremely rough in its initial commit

example lines of code...

Executing from the code directory:
rv_process_dir('PROCESSED_DATA_DIRECTORY')

fn = 'T2m3wr-20140617.144009-0167.p11.fits'
flux,sig,wave = read_and_find_star_p11(fn)

fn = 'T2m3wr-20140617.144009-0167.p08.fits'
flux,wave = read_and_find_star_p08(fn)
spectrum,sig = weighted_extract_spectrum(flux)
rv,rv_sig,temp = calc_rv_ambre(spectrum,wave,sig,'ambre_conv', ([0,5400],[6870,6890]))
"""

from __future__ import print_function
try:
    import pyfits
except:
    import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import pdb
import glob


def read_and_find_star_p11(fn, manual_click=False, npix=7, subtract_sky=True,sky_rad=2):
    """Read in a cube and find the star.
    Return a postage stamp around the star and the coordinates
    within the stamp
    
    NB This didn't really work as the details of flux calibration doesn't easily 
    enable optimal extraction. 
    
    This function should probably be REMOVED.
    """
    a = pyfits.open(fn)
    #Assume Stellar mode.
    flux = a[0].data[:,:,13:]
    sig = a[1].data[:,:,13:]
    image = np.median(flux,axis=0)
    maxpx = np.unravel_index(np.argmax(image[1:-1,1:-1]),image[1:-1,1:-1].shape)
    maxpx = (maxpx[0]+1,maxpx[1]+1)
    plt.clf()
    plt.imshow(image,interpolation='nearest')
    plt.plot(maxpx[1],maxpx[0],'wx')
    if subtract_sky:
        xy = np.meshgrid(range(image.shape[1]),range(image.shape[0]))
        dist = np.sqrt((xy[0]-maxpx[1])**2.0 + (xy[1]-maxpx[0])**2.0)
        sky = np.where( (xy[0] > 0) & (xy[1] > 0) & 
            (xy[0] < image.shape[1]-1) & (xy[1] < image.shape[0]-1) &
            (dist > sky_rad) & (dist < image.shape[1]))
        for i in range(flux.shape[0]):
            flux[i,:,:] -= np.median(flux[i,sky[0],sky[1]])
    ymin = np.min([np.max([maxpx[0]-3,0]),image.shape[0]-npix])
    xmin = np.min([np.max([maxpx[1]-3,0]),image.shape[1]-npix])
    flux_stamp = flux[:,ymin:ymin+npix,xmin:xmin+npix]
    sig_stamp = sig[:,ymin:ymin+npix,xmin:xmin+npix]
    wave = a[0].header['CRVAL3'] + np.arange(flux.shape[0])*a[0].header['CDELT3']
    return flux_stamp,sig_stamp,wave
    
def read_and_find_star_p08(fn, manual_click=False, npix=7, subtract_sky=True,sky_rad=2,fig_fn=''):
    """Read in a cube and find the star.
    Return a postage stamp around the star and the wavelength scale
    
    NB This didn't really work as the details of flux calibration doesn't easily 
    enable optimal extraction.
    
    Parameters
    ----------
    fn: string
        filename
    npix: int
        Number of pixels to extract
    """
    a = pyfits.open(fn)
    #Assume Stellar mode.
    flux = np.array([a[i].data for i in range(1,13)])
    wave = a[1].header['CRVAL1'] + np.arange(flux.shape[2])*a[1].header['CDELT1']
    image = np.median(flux,axis=2)
    #!!! 1->7 is a HACK - because WiFeS seems to often fail on the edge pixels !!!
    maxpx = np.unravel_index(np.argmax(image[1:-1,7:-1]),image[1:-1,7:-1].shape)
    maxpx = (maxpx[0]+1,maxpx[1]+7)
    plt.clf()
    plt.imshow(image,interpolation='nearest')
    plt.plot(maxpx[1],maxpx[0],'wx')
    if subtract_sky:
        xy = np.meshgrid(range(image.shape[1]),range(image.shape[0]))
        dist = np.sqrt((xy[0]-maxpx[1])**2.0 + (xy[1]-maxpx[0])**2.0)
        sky = np.where( (xy[0] > 0) & (xy[1] > 0) & 
            (xy[0] < image.shape[1]-1) & (xy[1] < image.shape[0]-1) &
            (dist > sky_rad) & (dist < image.shape[1]))
        for i in range(flux.shape[2]):
            flux[:,:,i] -= np.median(flux[sky[0],sky[1],i])
    ymin = np.min([np.max([maxpx[0]-npix//2,0]),image.shape[0]-npix])
    xmin = np.min([np.max([maxpx[1]-npix//2,0]),image.shape[1]-npix])
    flux_stamp = flux[ymin:ymin+npix,xmin:xmin+npix,:]
    if len(fig_fn)>0:
        plt.savefig(fig_fn)
    return flux_stamp,wave
    
def weighted_extract_spectrum(flux_stamp, readout_var=11.0):
    """Optimally extract the spectrum based on a constant weighting
    
    Based on a p08 file axis ordering.
    
    Readout variance is roughly 11 in the p08 extracted spectra
    """
    flux_med = np.median(flux_stamp,axis=2)
    weights = flux_med/(flux_med + readout_var)
    spectrum = np.array([np.sum(flux_stamp[:,:,i]*weights) for i in range(flux_stamp.shape[2])])
    sig = np.array([np.sqrt(np.sum((flux_stamp[:,:,i]+readout_var)*weights**2)) for i in range(flux_stamp.shape[2])])
    return spectrum,sig
    
def conv_ambre_spect(ambre_dir,ambre_conv_dir):
    """Take all the AMBRE spectra from a directory, convolve and re-sample
    by a factor of 10, then save to a new directory"""
    infns = glob.glob(ambre_dir + '/*fits')
    for infn in infns:
        data = pyfits.getdata(infn)
        data = np.convolve(data,np.ones(10)/10., 'same')
        conv_data = data[10*np.arange(90000,dtype='int')].astype('float32')
        ix_start = infn.rfind('/') + 1
        ix_end = infn.rfind('.')
        outfn = infn[ix_start:ix_end] + 'conv.fits'
        pyfits.writeto(ambre_conv_dir + '/' + outfn,conv_data, clobber=True)
    
def rv_fit_mlnlike(shift,modft,data,errors,gaussian_offset):
    """Return minus the logarithm of the likelihood of the model fitting the data
    
    Parameters
    ----------
    shift: float
        Shift in pixels
    modft: array-like
        Real numpy Fourier transform of the model spectrum.
    data: array-like
        spectral data.
    errors: array-like
        uncertainties in spectral data
    gaussian_offset: float
        Offset to Gaussian uncertainty distribution
    """
    shifted_mod = np.fft.irfft(modft * np.exp(-2j * np.pi * np.arange(len(modft))/len(modft) * shift))
    return -np.sum(np.log(np.exp(-(data - shifted_mod)**2/2.0/errors**2) + gaussian_offset))

    
def calc_rv_ambre(spect,wave,sig, ambre_conv_dir,bad_intervals,smooth_distance=101, \
    gaussian_offset=1e-5,nwave_log=1e4,oversamp=1,fig_fn=''):
    """Compute a radial velocity based on an best fitting AMBRE spectrum.
    Teff is estimated at the same time.
    
    Parameters
    ----------
    spect: array-like
        The reduced WiFeS spectrum
        
    wave: array-like
        The wavelengths corresponding to the reduced WiFeS spectrum
        
    ambre_conv_dir: string
        The directory containing AMBRE spectra convolved to 0.1 Angstrom resolution
        
    bad_intervals: 
        List of wavelength intervals where e.g. telluric absorption is bad.
        
    smooth_distance: float
        Distance to smooth for "continuum" correction
        
    oversamp: float
        Oversampling of the input wavelength scale. The slit is assumed 2 pixels wide.
    
    frac_bad: float
        Fraction of bad spectral pixels assumed - goes into the likelihood function.
        
    Returns
    -------
    rv: float
        Radial velocity in km/s
    rv_sig: float
        Uncertainty in radial velocity (NB assumes good model fit)
    temp: int
        Temperature of model spectrum used for cross-correlation.
    """
    dell_ambre = 0.1
    wave_ambre=np.arange(90000)*dell_ambre + 3000
    wave_log = np.min(wave)*np.exp( np.log(np.max(wave)/np.min(wave))/nwave_log*np.arange(nwave_log))
    #Interpolate the spectrum onto this scale
    spect_int = np.interp(wave_log,wave,spect)
    #!!! Testing !!!
    #spect_int = np.roll(spect_int,+1) #redshift, positive RV
    sig_int = np.interp(wave_log,wave,sig)
    #Normalise 
    sig_int /= np.median(spect_int)
    spect_int /= np.median(spect_int)
    #Remove bad intervals 
    for interval in bad_intervals:
        wlo = np.where(wave_log > interval[0])[0]
        if len(wlo)==0: 
            continue
        whi = np.where(wave_log > interval[1])[0]
        if len(whi)==0:
            whi = [len(wave_log)]
        whi = whi[0]
        wlo = wlo[0]
        spect_int[wlo:whi] = spect_int[wlo] + np.arange(whi-wlo,dtype='float')/(whi-wlo)*(spect_int[whi] - spect_int[wlo])
        sig_int[wlo:whi]=1
    #Subtract smoothed spectrum
    spect_int -= spect_int[0] + np.arange(len(spect_int))/(len(spect_int)-1.0)*(spect_int[-1]-spect_int[0])
    spect_int -= np.convolve(spect_int,np.ones(smooth_distance)/smooth_distance,'same')
    ambre_fns = glob.glob(ambre_conv_dir + '/*.fits')
    rvs = np.zeros(len(ambre_fns))
    peaks = np.zeros(len(ambre_fns))
    ambre_ints = np.zeros( (len(ambre_fns),len(wave_log)) )
    drv = np.log(wave_log[1]/wave_log[0])*2.998e5
    for i,ambre_fn in enumerate(ambre_fns):
        #Smooth the ambre spectrum
        ambre_subsamp = int((wave[1]-wave[0])/dell_ambre)
        spect_ambre = pyfits.getdata(ambre_fn)
        spect_ambre = np.convolve(np.convolve(spect_ambre,np.ones(ambre_subsamp)/ambre_subsamp,'same'),\
                                  np.ones(2*ambre_subsamp)/ambre_subsamp/2,'same')
        #Interpolate onto the log wavelength grid.
        ambre_int = np.interp(wave_log,wave_ambre,spect_ambre)
        #Normalise 
        ambre_int /= np.median(ambre_int)
        #Remove bad intervals 
        for interval in bad_intervals:
            wlo = np.where(wave_log > interval[0])[0]
            if len(wlo)==0: 
                continue
            whi = np.where(wave_log > interval[1])[0]
            if len(whi)==0:
                whi = [len(wave_log)]
            whi = whi[0]
            wlo = wlo[0]
            ambre_int[wlo:whi] = ambre_int[wlo] + np.arange(whi-wlo)/(whi-wlo)*(ambre_int[whi] - ambre_int[wlo])
        #Subtract smoothed spectrum
        ambre_int -= ambre_int[0] + np.arange(len(ambre_int))/(len(ambre_int)-1)*(ambre_int[-1]-ambre_int[0])
        ambre_int -= np.convolve(ambre_int,np.ones(smooth_distance)/smooth_distance,'same')
        ambre_ints[i,:] =  ambre_int
        cor = np.correlate(spect_int,ambre_int,'same')
        peaks[i] = np.max(cor)/np.sum(np.abs(ambre_int)**0.5) #!!! Hack!!! 
        rvs[i] = (np.argmax(cor) - nwave_log/2)*drv 
    ix = np.argmax(peaks)
    #Recompute and plot the best cross-correlation
    ambre_int = ambre_ints[ix,:]
    cor = np.correlate(spect_int,ambre_int,'same')
    plt.clf()
    plt.plot(drv*(np.arange(2*smooth_distance)-smooth_distance), cor[nwave_log/2-smooth_distance:nwave_log/2+smooth_distance])
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('X Correlation')
    fn_ix = ambre_fns[ix].rfind('/')
    temperature = ambre_fns[ix][fn_ix+2:fn_ix+6]

    #Fit for a precise RV... note that minimize (rather than minimize_scalar) failed more
    #often for spectra that were not good matches.
    modft = np.fft.rfft(ambre_int)
    #res = op.minimize(rv_fit_mlnlike,rvs[ix]/drv,args=(modft,spect_int,sig_int,gaussian_offset))
    #x = res.x[0]
    res = op.minimize_scalar(rv_fit_mlnlike,args=(modft,spect_int,sig_int,gaussian_offset),bounds=((rvs[ix]-1)/drv,(rvs[ix]+1)/drv))
    x = res.x
    rv = x*drv
    fplus = rv_fit_mlnlike(x+0.5,modft,spect_int,sig_int,gaussian_offset)
    fminus = rv_fit_mlnlike(x-0.5,modft,spect_int,sig_int,gaussian_offset)
    hess_inv = 0.5**2/(fplus +  fminus - 2*res.fun)
    if hess_inv < 0:
        #If you get here, then there is a problem with the input spectrum or fitting.
        #raise UserWarning
        print("WARNING: Radial velocity fit did not work!")
    rv_sig = np.sqrt(hess_inv*nwave_log/len(spect)/oversamp)*drv
    plt.title('T = ' + temperature + ' K, RV = {0:4.1f}+/-{1:4.1f} km/s'.format(rv,rv_sig))
    if len(fig_fn) > 0:
        plt.savefig(fig_fn)
    return rv,rv_sig,int(temperature)
    
def rv_process_dir(dir,ambre_conv_dir='./ambre_conv/',standards_dir='',outfn='rvs.txt'):
    """Process all files in a directory for radial velocities.
    
    Parameters
    ----------
    dir: string
        Directory in which to process the WiFeS reduced spectra
    ambre_conf_dir: string
        Directory containing AMBRE spectra convolved to WiFeS resolution
    outfn: string
        Output filename"""
    if len(standards_dir)>0:
        print("WARNING: Feature not implemented yet")
        raise UserWarning
    fns = glob.glob(dir + '/*p08.fits'  )
    outfile = open(dir + '/' + outfn,'w')
    for fn in fns:
        h = pyfits.getheader(fn)
        flux,wave = read_and_find_star_p08(fn,fig_fn=dir + h['OBJNAME'] + '.' + h['OBSID'] + '_star.png')
        if h['BEAMSPLT']=='RT560':
            bad_intervals = ([0,5500],[6865,6935],)
        else:
            bad_intervals = ([6865,6935],)
        spectrum,sig = weighted_extract_spectrum(flux)
        rv,rv_sig,temp = calc_rv_ambre(spectrum,wave,sig,ambre_conv_dir, bad_intervals,\
            fig_fn=dir + h['OBJNAME'] + '.' + h['OBSID'] + '_xcor.png')
        rv += h['RADVEL']
        outfile.write(h['OBJNAME'] + ' & '+ h['RA'] + ' & '+ h['DEC'] + ' & ' + h['BEAMSPLT'] + \
            ' & {0:10.3f} & {1:5.1f} $\pm$ {2:5.1f} & {3:5.0f} \\\\ \n'.format(h['MJD-OBS'],rv,rv_sig,temp))
    outfile.close()
    