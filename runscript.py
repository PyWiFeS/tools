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
rv_process_dir('/Volumes/UTRAID/wifes_data/140623/reduction_red_150806',
template_conv_dir='./phoenix_conv/',outdir='arizz_outputs/phoenix_outputs')

