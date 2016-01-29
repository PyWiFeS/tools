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
do_outtex=0

data = read_csv('arizz_outputs/140623/phoenix/rvs.txt',string='name,filename,bmsplt,ra,dec')
outtex = 'rv_140623_phoenix.tex'

;data = read_csv('arizz_outputs/140622/phoenix/rvs.txt',string='name,filename,bmsplt,ra,dec')
;outtex = 'rv_140622_phoenix.tex'

;data = read_csv('arizz_outputs/140621/phoenix/rvs.txt',string='name,filename,bmsplt,ra,dec')
;outtex = 'rv_140621_phoenix.tex'

;data = read_csv('arizz_outputs/140619/phoenix/rvs.txt',string='name,filename,bmsplt,ra,dec')
;outtex = 'rv_140619_phoenix.tex'


;;outfile = 'rv_140622_phoenix.txt'
;outtex = 'rv_140622_phoenix.tex'

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
;xrv = xrv[1:*]

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

