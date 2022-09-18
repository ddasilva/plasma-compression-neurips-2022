import numpy as np


def trap_integ(x,y):
    """Trapezoidal integration function.
    
    Args
       x: x-coordinates
       y: function values at x-coordinates
    Returns
       integral of function f(x[i]) = y[i] using trapezoidal integration
    """
    ndata = len(x)
    dx = x[1:ndata]-x[0:(ndata-1)]
    ys = y[1:ndata]+y[0:(ndata-1)]
    return 0.5*np.sum(dx*ys)


def idl_moments(psd,energies,theta,phi,sensor='dis', E0=100.,integrands=0,Smat=None, extras=False):
    """Moments integration code converted from the MMS/FPI ground system.
    
    Args
       psd: phase space density
       energies: energies (in eV)
       theta: elevation angle (from nadir)
       phi: azimuth
       integrands: do not use
       Smat: do not use
    Returns
       dictionary containing keys n, vx, vy, vz, txx, tyy, tzz, txy, txz, tyz.
    """
    theta_ext = np.hstack((0,theta,180.))*np.pi/180. # add 0 and 180 deg values to theta array
    phi_ext = np.hstack((phi,phi[-1]+11.25))*np.pi/180. # extend phi array to loop around in azimuth
 
    psd_proc = np.swapaxes(psd,0,2)
    fgrid = np.zeros((psd_proc.shape[0]+2,psd_proc.shape[1]+2,psd_proc.shape[2]+1),np.float)
 
    fgrid[1:-1,1:-1,0:-1] = psd_proc
    fgrid[1:-1,0,0:-1] = 0.
    fgrid[1:-1,-1,0:-1] = 0.
    fgrid[1:-1,1:-1,-1] = psd_proc[:,:,0]  
 
    nen = fgrid.shape[0]
    nth = fgrid.shape[1]
 
    if sensor == 'des':
        mass = 9.10938291e-28        
    else:
        mass = 1.67262178e-24
    
    kb =  1.3807e-16
    temp1eV =  1.1604e4
    energy1eV = 1.60217657e-12
 
    intphi0 = np.zeros((nen, nth), np.float64)
    for i in range(nen):
        for j in range(nth):
            intphi0[i,j] = trap_integ(phi_ext,fgrid[i,j,:])
 
    intphi1 = np.zeros((nen, nth), np.float)
    for i in range(nen):
        for j in range(nth):
            intphi1[i,j] = trap_integ(phi_ext,fgrid[i,j,:]*np.cos(phi_ext))
 
    intphi2 = np.zeros((nen, nth), np.float)
    for i in range(nen):
        for j in range(nth):
            intphi2[i,j] = trap_integ(phi_ext,fgrid[i,j,:]*np.sin(phi_ext))
 
    intphi3 = np.zeros((nen,nth), np.float)
    for i in range(nen):
        for j in range(nth):
            intphi3[i,j] = trap_integ(phi_ext,fgrid[i,j,:]*np.cos(phi_ext)**2)
 
    intphi4 = np.zeros((nen, nth), np.float)
    for i in range(nen):
        for j in range(nth):
            intphi4[i,j] = trap_integ(phi_ext,fgrid[i,j,:]*np.sin(phi_ext)**2)
 
    intphi5 = np.zeros((nen, nth), np.float)
    for i in range(nen):
        for j in range(nth):
            intphi5[i,j] = trap_integ(phi_ext,fgrid[i,j,:]*np.sin(phi_ext)*np.cos(phi_ext))
 
    inttheta0 = np.zeros(nen, np.float)
    for i in range(nen):
        inttheta0[i] = trap_integ(theta_ext,intphi0[i,:]*np.sin(theta_ext))
 
    inttheta1 = np.zeros(nen, np.float)
    for i in range(nen):
        inttheta1[i] = -trap_integ(theta_ext,intphi1[i,:]*np.sin(theta_ext)**2)
 
    inttheta2 = np.zeros(nen, np.float)
    for i in range(nen):
        inttheta2[i] = -trap_integ(theta_ext,intphi2[i,:]*np.sin(theta_ext)**2)
 
    inttheta3 = np.zeros(nen, np.float)
    for i in range(nen):
        inttheta3[i] = -trap_integ(theta_ext,intphi0[i,:]*np.sin(theta_ext)*np.cos(theta_ext))
 
    inttheta4 = np.zeros(nen, np.float)
    for i in range(nen):
        inttheta4[i] = trap_integ(theta_ext,intphi3[i,:]*np.sin(theta_ext)**3)
 
    inttheta5 = np.zeros(nen, np.float)
    for i in range(nen):
        inttheta5[i] = trap_integ(theta_ext,intphi4[i,:]*np.sin(theta_ext)**3)
 
    inttheta6 = np.zeros(nen, np.float)
    for i in range(nen):
        inttheta6[i] = trap_integ(theta_ext,intphi0[i,:]*np.sin(theta_ext)*np.cos(theta_ext)**2)
 
    inttheta7 = np.zeros(nen, np.float)
    for i in range(nen):
        inttheta7[i] = trap_integ(theta_ext,intphi5[i,:]*np.sin(theta_ext)**3)
 
    inttheta8 = np.zeros(nen, np.float)
    for i in range(nen):
        inttheta8[i] = trap_integ(theta_ext,intphi1[i,:]*np.sin(theta_ext)**2*np.cos(theta_ext))
 
    inttheta9 = np.zeros(nen, np.float)
    for i in range(nen):
        inttheta9[i] = trap_integ(theta_ext,intphi2[i,:]*np.sin(theta_ext)**2*np.cos(theta_ext))
 
    E0_cgs = E0*energy1eV
    
    integrand_n = np.zeros(nen, np.float)
    integrand_nvx = np.zeros(nen, np.float)
    integrand_nvy = np.zeros(nen, np.float)
    integrand_nvz = np.zeros(nen, np.float)
    integrand_ntxx = np.zeros(nen, np.float)
    integrand_ntyy = np.zeros(nen, np.float)
    integrand_ntzz = np.zeros(nen, np.float)
    integrand_ntxy = np.zeros(nen, np.float)
    integrand_ntxz = np.zeros(nen, np.float)
    integrand_ntyz = np.zeros(nen, np.float)  
    
    ugrid = np.zeros(nen, np.float)
    
    ugrid[0] = 0.
    ugrid[-1] = 1.
    ugrid[1:-1] = energies/(E0+energies)
  
    integrand_n[1:-1] = \
            np.sqrt(2.)/(mass)**1.5*(E0_cgs)**1.5*np.sqrt(ugrid[1:-1])/(1.-ugrid[1:-1])**2.5*inttheta0[1:-1]
 
    integrand_nvx[1:-1] = \
        2./(mass**2)*E0_cgs**2*ugrid[1:-1]/(1.-ugrid[1:-1])**3*inttheta1[1:-1]
    
    integrand_nvy[1:-1] = \
        2./(mass**2)*E0_cgs**2*ugrid[1:-1]/(1.-ugrid[1:-1])**3*inttheta2[1:-1]
 
    integrand_nvz[1:-1] = \
        2./(mass**2)*E0_cgs**2*ugrid[1:-1]/(1.-ugrid[1:-1])**3*inttheta3[1:-1]
 
    integrand_ntxx[1:-1] = \
        2.**(1.5)/mass**(2.5)*E0_cgs**(2.5)*ugrid[1:-1]**(1.5)/(1.-ugrid[1:-1])**(3.5)*inttheta4[1:-1]
 
    integrand_ntyy[1:-1] = \
        2.**(1.5)/mass**(2.5)*E0_cgs**(2.5)*ugrid[1:-1]**(1.5)/(1.-ugrid[1:-1])**(3.5)*inttheta5[1:-1]
 
    integrand_ntzz[1:-1] = \
        2.**(1.5)/mass**(2.5)*E0_cgs**(2.5)*ugrid[1:-1]**(1.5)/(1.-ugrid[1:-1])**(3.5)*inttheta6[1:-1]
 
    integrand_ntxy[1:-1] = \
        2.**(1.5)/mass**(2.5)*E0_cgs**(2.5)*ugrid[1:-1]**(1.5)/(1.-ugrid[1:-1])**(3.5)*inttheta7[1:-1]
 
    integrand_ntxz[1:-1] = \
        2.**(1.5)/mass**(2.5)*E0_cgs**(2.5)*ugrid[1:-1]**(1.5)/(1.-ugrid[1:-1])**(3.5)*inttheta8[1:-1]
 
    integrand_ntyz[1:-1] = \
        2.**(1.5)/mass**(2.5)*E0_cgs**(2.5)*ugrid[1:-1]**(1.5)/(1.-ugrid[1:-1])**(3.5)*inttheta9[1:-1]
 
 
    if integrands == 0:
        n = trap_integ(ugrid,integrand_n)
        vx = trap_integ(ugrid,integrand_nvx)/1e5 / n
        vy = trap_integ(ugrid,integrand_nvy)/1e5 / n
        vz = trap_integ(ugrid,integrand_nvz)/1e5 / n
        txx = mass*trap_integ(ugrid,integrand_ntxx)/n/kb/temp1eV - 1e10*mass*vx**2/kb/temp1eV
        tyy = mass*trap_integ(ugrid,integrand_ntyy)/n/kb/temp1eV - 1e10*mass*vy**2/kb/temp1eV
        tzz = mass*trap_integ(ugrid,integrand_ntzz)/n/kb/temp1eV - 1e10*mass*vz**2/kb/temp1eV
        txy = mass*trap_integ(ugrid,integrand_ntxy)/n/kb/temp1eV - 1e10*mass*vx*vy/kb/temp1eV
        txz = mass*trap_integ(ugrid,integrand_ntxz)/n/kb/temp1eV - 1e10*mass*vx*vz/kb/temp1eV
        tyz = mass*trap_integ(ugrid,integrand_ntyz)/n/kb/temp1eV - 1e10*mass*vy*vz/kb/temp1eV

        if Smat != None:
            Sinv = np.linalg.inv(Smat)
 
            Tmat = np.array([[txx,txy,txz],[txy,tyy,tyz],[txz,tyz,tzz]])               
            Tb = np.dot(np.dot(Sinv,Tmat),Smat)
            
            txx = Tb[0,0]
            tyy = Tb[1,1]
            tzz = Tb[2,2]
            txy = Tb[0,1]
            txz = Tb[0,2]
            tyz = Tb[1,2]   
            
    else:
        n = integrand_n
        vx = integrand_nvx/1e5 / trap_integ(ugrid,integrand_n)
        vy = integrand_nvy/1e5 / trap_integ(ugrid,integrand_n)
        vz = integrand_nvz/1e5 / trap_integ(ugrid,integrand_n)
        txx = integrand_ntxx / trap_integ(ugrid,integrand_n) * mass/kb/temp1eV
        tyy = integrand_ntyy / trap_integ(ugrid,integrand_n) * mass/kb/temp1eV
        tzz = integrand_ntzz / trap_integ(ugrid,integrand_n) * mass/kb/temp1eV
        txy = integrand_ntxy / trap_integ(ugrid,integrand_n) * mass/kb/temp1eV
        txz = integrand_ntxz / trap_integ(ugrid,integrand_n) * mass/kb/temp1eV
        tyz = integrand_ntyz / trap_integ(ugrid,integrand_n) * mass/kb/temp1eV
    
        if Smat != None:
            Sinv = np.linalg.inv(Smat)
        
            for tii in range(txx.size):
                Tmat = np.array([[txx[tii],txy[tii],txz[tii]],[txy[tii],tyy[tii],tyz[tii]],[txz[tii],tyz[tii],tzz[tii]]])
            
                Tb = np.dot(np.dot(Sinv,Tmat),Smat)
                txx[tii] = Tb[0,0]
                tyy[tii] = Tb[1,1]
                tzz[tii] = Tb[2,2]
                txy[tii] = Tb[0,1]
                txz[tii] = Tb[0,2]
                tyz[tii] = Tb[1,2]
    
    result = dict(n=n,vx=vx,vy=vy,vz=vz,txx=txx,tyy=tyy,tzz=tzz,txy=txy,txz=txz,tyz=tyz)

    
    if extras:
        result['nvx'] = trap_integ(ugrid,integrand_nvx)
        result['nvy'] = trap_integ(ugrid,integrand_nvy)
        result['nvz'] = trap_integ(ugrid,integrand_nvz)

        result['ntxx'] = trap_integ(ugrid,integrand_ntxx)
        result['ntyy'] = trap_integ(ugrid,integrand_ntyy)
        result['ntzz'] = trap_integ(ugrid,integrand_ntzz)
        result['ntxy'] = trap_integ(ugrid,integrand_ntxy)
        result['ntxz'] = trap_integ(ugrid,integrand_ntxz)
        result['ntyz'] = trap_integ(ugrid,integrand_ntyz)
        
    return result