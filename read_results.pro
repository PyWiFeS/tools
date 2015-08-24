;;#star,filename,ra,dec,bmsplt,mjd,rv,sig_rv,teff
data = read_csv('arizz_outputs/rvs.txt',string='name,filename,bmsplt,ra,dec')

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
;;xrv = xrv[1,*]

;;derive the radial velocity calibration in the simplest way possible
myfun = 'X+p[0]'
rv_offset = mpfitexpr(myfun,rvx[xrv],data.rv[xrv],data.sig_rv[xrv],[0.0],dof=dof,bestnorm=cs,perror=perror)
rv_offerr = sqrt(cs/dof)*perror
print, 'RV Offset: ' + nform(rv_offset,dec=3)+' +- ' + nform(rv_offerr,dec=3) 


;;correct the rvs and add the correction error in quadrature
rvs = data.rv+rv_offset[0]
sig_rvs = sqrt(data.sig_rv^2 + rv_offerr[0]^2)


stop
end

