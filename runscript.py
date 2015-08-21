import process_stellar
import matplotlib.pyplot as plt
conv_tlusty_spect = process_stellar.conv_tlusty_spect
conv_phoenix_spect = process_stellar.conv_phoenix_spect
rv_process_dir = process_stellar.rv_process_dir

##CONVOLVING TEMPLATES
#conv_tlusty_spect('/Volumes/UTRAID/TLUSTY/BGvispec_v2/','tlusty_conv')
#conv_phoenix_spect('/Volumes/UTRAID/phoenix_hires/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/foruse/','phoenix_conv')


##RUNNING RV FITTER ON DATA
##Executing from the code directory:
rv_process_dir('/Volumes/UTRAID/wifes_data/140623/reduction_red_150806',template_conv_dir='./phoenix_conv/',outdir='arizz_outputs')

#fn = 'T2m3wr-20140617.144009-0167.p11.fits'
#flux,sig,wave = read_and_find_star_p11(fn)

#fn = 'T2m3wr-20140617.144009-0167.p08.fits'
#flux,wave = read_and_find_star_p08(fn)
#spectrum,sig = weighted_extract_spectrum(flux)
#rv,rv_sig,temp = calc_rv_templates(spectrum,wave,sig,'template_conv', ([0,5400],[6870,6890]))