"""
To run, type:

%run process_stellar
%run test_process_stellar
"""
from __future__ import print_function
try:
    import pyfits
except:
    import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
plt.ion()

mode='plot' #fit, plot or test

#From Margaret:
#They were all obtained with 'T2m3wb-20141110.113949-0831.p08.fits', using RV_templates/8500g40p00k2v50.txt and 5000g35p00k2v50.txt and alpha =0.1.

if mode=='fit':
    files = glob.glob('C:/Users/Margaret/MPhil/rvs_p08files_TTHor_2016/*.fits')
    base_path = 'C:/Users/Margaret/MPhil/rvs_p08files_TTHor_2016/'
    for file in files:  
        file_num = int(file.split('.')[1].split('-')[1])
        flux,wave = read_and_find_star_p08(file)
        spectrum,sig = weighted_extract_spectrum(flux)
        helcor = pyfits.getheader(file)['RADVEL']
        wave_log, spect_int, model_spect = calc_rv_todcor(spectrum,wave,sig,\
            ['C:/Users/Margaret/MPhil/RV_templates/8500g40p00k2v50.txt',\
            'C:/Users/Margaret/MPhil/RV_templates/5000g35p00k2v50.txt'],\
            alpha=0.1,out_fn='rvs.txt',jd=file_num,return_fitted=True,\
            heliocentric_correction=helcor)

        plt.clf()
        plt.plot(wave_log, spect_int, label='Data')
        plt.plot(wave_log, model_spect, label='Model')
        plt.legend()
        plt.xlabel('Wavelength')
elif mode=='plot':
    #** To test the flux normalization of todcor **
    templates_dir = '/Users/mireland/python/pywifes/tools/margaret/'
    template_fns = glob.glob(templates_dir+'*txt')
    plot_p08_dir = '/Users/mireland/data/wifes/rvs_p08files/'
    plot_p08_dir = '/Users/mireland/python/pywifes/tools/margaret/'

    flux, wave = read_and_find_star_p08(plot_p08_dir + 'T2m3wb-20141110.113949-0831.p08.fits')
    spectrum, sig = weighted_extract_spectrum(flux)
    dummy = calc_rv_todcor(spectrum, wave,sig, [template_fns[3], template_fns[1]], bad_intervals=[[0,3810], [5005, 5028]], alpha=0.25, plotit=True)
    #dummy = calc_rv_todcor(spectrum, wave,sig, [template_fns[3], template_fns[1]], bad_intervals=[[0,4200], [5067,5075],[5500,6000]], alpha=0.25, plotit=True)
    #dummy = calc_rv_todcor(spectrum, wave,sig, [template_fns[2], template_fns[1]], bad_intervals=[[0,4200], [5067,5075],[5500,6000]], alpha=0.25, plotit=True)
elif mode=='test':
    #*** lines below test todcor ***
    binspect,binwave,binsig=make_fake_binary(spectrum, wave, sig, ['RV_templates/9000g40p00k2v150.txt','RV_templates/5250g35p00k2v150.txt'],0.5,-200,+200)
    calc_rv_todcor(binspect,binwave,binsig,['RV_templates/9000g40p00k2v150.txt','RV_templates/5250g35p00k2v150.txt'],alpha=0.5)

    rv,rv_sig = calc_rv_template(spectrum,wave,sig,'template_conv', ([0,5400],[6870,6890]))
    rv,rv_sig = calc_rv_template(spectrum,wave,sig,template_fns, ([0,5400],[6870,6890]))

