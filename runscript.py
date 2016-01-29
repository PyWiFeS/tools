import process_stellar
import matplotlib.pyplot as plt
conv_tlusty_spect = process_stellar.conv_tlusty_spect
conv_phoenix_spect = process_stellar.conv_phoenix_spect
rv_process_dir = process_stellar.rv_process_dir
import pdb
import time
import numpy as np


##CONVOLVING TEMPLATES
#conv_tlusty_spect('/Volumes/UTRAID/TLUSTY/BGvispec_v2/','tlusty_conv')
#conv_phoenix_spect('/Volumes/UTRAID/phoenix_hires/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/foruse/','phoenix_conv')


##RUNNING RV FITTER ON DATA
##Executing from the code directory:
#rv_process_dir('/Volumes/UTRAID/wifes_data/140619/reduction_red_150806',
#template_conv_dir='./phoenix_conv/',outdir='arizz_outputs/140619/phoenix',mask_ha_emission=True)

#rv_process_dir('/Users/arizz/python/pywifes/tools/test_intput',
#template_conv_dir='/Users/arizz/python/pywifes/tools/full_conv/',outdir='/Users/arizz/python/pywifes/tools/testing_outputs',mask_ha_emission=False)

indirs = np.array(['140623','140622','140621','140619'])

for ii in indirs:
    indir= '/Volumes/UTRAID/wifes_data/'+ii+'/reduction_red_150806'
    odir = 'arizz_outputs/'+indir.split('/')[4]+'/both'
    #pdb.set_trace()
    rv_process_dir(indir,template_conv_dir='/Users/arizz/python/pywifes/tools/full_conv/',outdir=odir,mask_ha_emission=False)
