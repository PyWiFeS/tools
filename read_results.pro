;;#star,filename,ra,dec,bmsplt,mjd,rv,sig_rv,teff
;datap = read_csv('arizz_outputs/140623/phoenix_outputs/rvs.txt',string='name,filename,bmsplt,ra,dec')
;datat = read_csv('arizz_outputs/140623/tlusty_outputs/rvs.txt',string='name,filename,bmsplt,ra,dec')
;stop
;plot, datap.rv,datat.rv,psym=4
;oplot,[-1000,1000],[-1000,1000]
;qwe = where(abs(datap.rv-datat.rv) gt 10)
;stop
;;pulkovo catalog of radial velocities for HIP stars
restore, '~/data_local/catalogs/pulkovo_rv/pulkovo.idlsave'
restore, '~/data_local/catalogs/hip_main.dat'
do_outtex=0
do_outcsv=1

;data = read_csv('arizz_outputs/140623/phoenix/rvs.txt',string='name,filename,bmsplt,ra,dec')
;outtex = 'arizz_outputs/outputs/rv_140623_hip.tex'
;outcsv = 'arizz_outputs/outputs/rv_140623_hip.csv'
;restore, '~/code/wifes_dr/afterpython/reduction_outputs/ewinspect_140623_specdat_140823_sptsnipe.idlsave'
;restore, '~/code/wifes_dr/afterpython/reduction_outputs/ewinspect_140623_specdat_140823.idlsave'

;data = read_csv('arizz_outputs/140622/phoenix/rvs.txt',string='name,filename,bmsplt,ra,dec')
;outtex = 'rv_140622_hip.tex'
;outcsv = 'arizz_outputs/outputs/rv_140622_hip.csv'
;restore, '~/code/wifes_dr/afterpython/reduction_outputs/ewinspect_140622_specdat_140822_sptsnipe.idlsave'
;restore, '~/code/wifes_dr/afterpython/reduction_outputs/ewinspect_140622_specdat_140822.idlsave'

;stop

;data = read_csv('arizz_outputs/140621/phoenix/rvs.txt',string='name,filename,bmsplt,ra,dec')
;outtex = 'rv_140621_phoenix.tex'
;outcsv = 'arizz_outputs/outputs/rv_140621_hip.csv'
;restore, '~/code/wifes_dr/afterpython/reduction_outputs/ewinspect_140621_specdat_140821_sptsnipe.idlsave'
;restore, '~/code/wifes_dr/afterpython/reduction_outputs/ewinspect_140621_specdat_140821.idlsave'
 

data = read_csv('arizz_outputs/140619/phoenix/rvs.txt',string='name,filename,bmsplt,ra,dec')
outtex = 'rv_140619_phoenix.tex'
outcsv = 'arizz_outputs/outputs/rv_140619_hip.csv'
restore, '~/code/wifes_dr/afterpython/reduction_outputs/ewinspect_140619_specdat_140816_sptsnipe.idlsave'
restore, '~/code/wifes_dr/afterpython/reduction_outputs/ewinspect_140619_specdat_140816.idlsave'

spts = out

rvstan = read_csv('rv_standards.txt',string='name')

rvx  = fltarr(n_elements(data.name))-1
sig_rvx = rvx*0.0-999.999
for i=0,n_elements(data.name)-1 do begin
   match = where(rvstan.name eq data.name[i])
   if match[0] ne -1 then begin
      rvx[i] = rvstan.rv[match]
      sig_rvx[i] = rvstan.sig_rv[match]
   endif
endfor
xrv = where(sig_rvx gt 0)
;;exclude an RV standard here if there is a good reason to do so:
;;xrv = xrv[1:*]

;;derive the radial velocity calibration in the simplest way possible
myfun = 'X+p[0]'
;stop
fitsig = sqrt(sig_rvx[xrv]^2 + data.sig_rv[xrv]^2)
;fitsig = data.sig_rv[xrv]
rv_offset = mpfitexpr(myfun,rvx[xrv],data.rv[xrv],fitsig,$
                      [0.0],dof=dof,bestnorm=cs,perror=perror)

rv_offerr = sqrt(cs/dof)*perror
print, 'RV Offset: ' + nform(rv_offset,dec=3)+' +- ' + nform(rv_offerr,dec=3) 


;;correct the rvs and add the correction error in quadrature
;;minus because we were fitting on the opposit direction above
rvs = data.rv-rv_offset[0]
sig_rvs = data.sig_rv + rv_offerr[0]


;;cross match everything in Pulkovo with the data
rvp     = fltarr(n_elements(data.name))-9999
sig_rvp = rvp

for i=0,n_elements(data.name)-1 do begin
   if strmid(data.name[i],0,3) eq 'HIP' then begin
      thiship = fix((strsplit(data.name[i],' ',/extract))[1],type=3)
      match = where(catdat.HIP eq thiship)
      if n_elements(match) gt 1 then stop
      if match[0] ne -1 then begin
         rvp[i] = catdat[match].rv
         sig_rvp[i] = catdat[match].sig_rv
      endif
   endif
endfor

pm = where(rvp gt -9998)
qwe = where(strmid(data.name,0,3) eq 'HIP')
sorter = sort(data.name[qwe])
stop

out = where(strmid(data.name,0,3) eq 'HIP')
numxrv = nform(n_elements(xrv))
if do_outcsv eq 1 then begin
   close,/all
   openw,1,outcsv
   printf,1,'HIP,RA,DEC,SpT(HD),SpT(WiFeS),rv,sig_rv,rv_off,sig_rv_off,n_rvstan,crv,sig_crv'
   for i=0,n_elements(out)-1 do begin
      thiship = fix(strmid(data.name[out[i]],3,10),type=3)
      xhip = where(hip_main.hip eq thiship)
      if xhip[0] eq -1 then stop
      hdcat = queryvizier('III/135A/catalog','HIP_'+nform(thiship,decimals=0),1.0,/allcolumns)
      if size(hdcat,/type) eq 8 then hashd=1 else hashd=0
      hdspt=''
      if hashd eq 1 then begin
         if n_elements(hdcat) gt 1 then stop
         hdspt = hdcat.spt
      endif

      gcirc,2,hip_main[xhip].alf*180/!PI,hip_main[xhip].del*180/!Pi,spec_dat.ra,spec_dat.dec,dist
      smatch = where(dist eq min(dist))
      if n_elements(smatch) gt 1 then stop
      if smatch[0] eq -1 then stop
      thisspt = spts[smatch].spts
      printf,1,nform(thiship,dec=0)+','+nform(hip_main[xhip].alf*180/!pi,dec=6)+','+nform(hip_main[xhip].del*180/!pi,dec=6)+','+ hdspt+','+strupcase(thisspt)+','+nform(data.rv[out[i]],dec=1)+','+nform(data.sig_rv[out[i]],dec=1)+','+nform(rv_offset,dec=3)+','+nform(rv_offerr,dec=3)+','+numxrv+','+nform(rvs[out[i]],dec=1) +','+nform(sig_rvs[out[i]],dec=1)
    ;stop

   endfor
endif
close,1
stop




if do_outtex eq 1 then begin
;;Output to new files with the corrected RV
openw,1,outtex
printf,1,'\hline'
printf,1,'     &     & RV     & $\sigma_{RV}$ \\'
printf,1,'Star & MJD & (Km/s) & (km/s)        \\'
printf,1,'\hline'
printf,1,'\hline'
for i=0,n_elements(rvs)-1 do begin
   line = data.name[i] + ' & ' + nform(data.mjd[i],dec=0) + ' & ' + $
          nform(rvs[i],dec=1) +' & ' + nform(sig_rvs[i],dec=1) + ' \\'
   
   printf,1,line   
endfor
printf,1,'\hline'
close,1
endif


stop
end

