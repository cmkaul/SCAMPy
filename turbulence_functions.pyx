import numpy as np
cimport numpy as np
from libc.math cimport cbrt, sqrt, log, fabs,atan, exp, fmax, pow
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
include "parameters.pxi"


# Entrainment Rates

cdef entr_struct entr_detr_cloudy(double z, double z_half,  double zi, double wk, double w_halfk) nogil:
    cdef entr_struct _ret
    cdef double eps = 1.0 # to avoid division by zero when z = 0 or z_i

    # in cloud portion from Soares 2004
    if z_half >= zi :
        _ret.entr_w = 1.5e-3 #2e-3
        _ret.entr_sc = 1.5e-3
        _ret.detr_w = 3.0e-3
        _ret.detr_sc= 3.0e-3
    else:
        # I think I just made this up to give a smooth blend
        _ret.entr_sc = 1.5e-3 * (1.0 - log(z_half/zi))
        _ret.entr_w = 1.5e-3 * (1.0 - log(z/zi))
        _ret.detr_w =  0.0#(log(fmax(z,20.0)/(zi)) - log(20.0/(zi))) * 1e-3
        _ret.detr_sc = 0.0 # (log(fmax(z_half,20.0)/(zi)) - log(20.0/(zi))) * 1e-3

    return  _ret


cdef entr_struct entr_detr_dry(double z, double z_half, double zi, double wk, double w_halfk) nogil:
    cdef entr_struct _ret
    cdef double eps = 1.0 # to avoid division by zero when z = 0 or z_i
    # Following Soares 2004
    _ret.entr_sc = 0.5*(1.0/fmax(z_half,10.0)+ 1.0/fmax(zi-z_half,10.0)) #vkb/(z + 1.0e-3)
    _ret.entr_w = 0.5*(1.0/fmax(z,10.0)+ 1.0/fmax(zi-z,10.0)) #vkb/z_half
    _ret.detr_w = 0.0
    _ret.detr_sc = 0.0

    return  _ret

cdef entr_struct entr_detr_inverse_z(double z, double z_half,  double zi, double wk, double w_halfk) nogil:
    cdef:
        entr_struct _ret
        double er0_zmin = 10.0 # lower limit for z in computation of entrainment/detrainment rates
    _ret.entr_sc = vkb/fmax(z_half, er0_zmin)
    _ret.entr_w = vkb/fmax(z, er0_zmin)
    _ret.detr_sc= _ret.entr_sc
    _ret.detr_w = _ret.entr_w
    return _ret

cdef entr_struct entr_detr_inverse_w(double z, double z_half,  double zi, double wk, double w_halfk) nogil:
    cdef entr_struct _ret
    cdef double eps = 1.0 # to avoid division by zero when z = 0 or z_i

    # in cloud portion from Soares 2004
    if z_half >= zi :
        _ret.detr_w = 3.0e-3
        _ret.detr_sc= 3.0e-3
    else:
        # I think I just made this up to give a smooth blend
        _ret.detr_w =  0.0#(log(fmax(z,20.0)/(zi)) - log(20.0/(zi))) * 1e-3
        _ret.detr_sc = 0.0 # (log(fmax(z_half,20.0)/(zi)) - log(20.0/(zi))) * 1e-3
    _ret.entr_w = 1.0/(400.0 * fmax(wk,0.1)) #standard 400
    _ret.entr_sc = 1.0/(400.0 * fmax(w_halfk,0.1)) #sets baseline to avoid errors
    return  _ret


# convective velocity scale
cdef double get_wstar(double bflux, double zi ):
    return cbrt(fmax(bflux * zi, 0.0))

# BL height
cdef double get_inversion(double *theta_rho, double *u, double *v, double *z_half,
                          Py_ssize_t kmin, Py_ssize_t kmax, double Ri_bulk_crit):
    cdef:
        double theta_rho_b = theta_rho[kmin]
        double h, Ri_bulk=0.0, Ri_bulk_low
        Py_ssize_t k = kmin


    # test if we need to look at the free convective limit
    if (u[kmin] * u[kmin] + v[kmin] * v[kmin]) <= 0.01:
        with nogil:
            for k in xrange(kmin,kmax):
                if theta_rho[k] > theta_rho_b:
                    break
        h = (z_half[k] - z_half[k-1])/(theta_rho[k] - theta_rho[k-1]) * (theta_rho_b - theta_rho[k-1]) + z_half[k-1]
    else:
        with nogil:
            for k in xrange(kmin,kmax):
                Ri_bulk_low = Ri_bulk
                Ri_bulk = g * (theta_rho[k] - theta_rho_b) * z_half[k]/theta_rho_b / (u[k] * u[k] + v[k] * v[k])
                if Ri_bulk > Ri_bulk_crit:
                    break
        h = (z_half[k] - z_half[k-1])/(Ri_bulk - Ri_bulk_low) * (Ri_bulk_crit - Ri_bulk_low) + z_half[k-1]

    return h

# MO scaling of near surface tke and scalar variance

cdef double get_surface_tke(double ustar, double wstar, double zLL, double oblength) nogil:
    if oblength < 0.0:
        return (3.75 + cbrt(zLL/oblength * zLL/oblength)) * ustar * ustar + 0.2 * wstar * wstar
    else:
        return 3.75 * ustar * ustar

cdef double get_surface_variance(double flux1, double flux2, double ustar, double zLL, double oblength) nogil:
    cdef:
        double c_star1 = -flux1/ustar
        double c_star2 = -flux2/ustar
    if oblength < 0.0:
        return 4.0 * c_star1 * c_star2 * pow(1.0 - 8.3 * zLL/oblength, -2.0/3.0)
    else:
        return 4.0 * c_star1 * c_star2



# Math-y stuff
cdef void construct_tridiag_diffusion(Py_ssize_t nzg, Py_ssize_t gw, double dzi, double dt,
                                 double *rho_ae_K_m, double *rho, double *ae, double *a, double *b, double *c):
    cdef:
        Py_ssize_t k
        double X, Y, Z #
        Py_ssize_t nz = nzg - 2* gw
    with nogil:
        for k in xrange(gw,nzg-gw):
            X = rho[k] * ae[k]/dt
            Y = rho_ae_K_m[k] * dzi * dzi
            Z = rho_ae_K_m[k-1] * dzi * dzi
            if k == gw:
                Z = 0.0
            elif k == nzg-gw-1:
                Y = 0.0
            a[k-gw] = - Z/X
            b[k-gw] = 1.0 + Y/X + Z/X
            c[k-gw] = -Y/X

    return


cdef void construct_tridiag_diffusion_implicitMF(Py_ssize_t nzg, Py_ssize_t gw, double dzi, double dt,
                                 double *rho_ae_K_m, double *massflux, double *rho, double *alpha, double *ae, double *a, double *b, double *c):
    cdef:
        Py_ssize_t k
        double X, Y, Z #
        Py_ssize_t nz = nzg - 2* gw
    with nogil:
        for k in xrange(gw,nzg-gw):
            X = rho[k] * ae[k]/dt
            Y = rho_ae_K_m[k] * dzi * dzi
            Z = rho_ae_K_m[k-1] * dzi * dzi
            if k == gw:
                Z = 0.0
            elif k == nzg-gw-1:
                Y = 0.0
            a[k-gw] = - Z/X + 0.5 * massflux[k-1] * dt * dzi/rho[k]
            b[k-gw] = 1.0 + Y/X + Z/X + 0.5 * dt * dzi * (massflux[k-1]-massflux[k])/rho[k]
            c[k-gw] = -Y/X - 0.5 * dt * dzi * massflux[k]/rho[k]

    return




cdef void construct_tridiag_diffusion_dirichlet(Py_ssize_t nzg, Py_ssize_t gw, double dzi, double dt,
                                 double *rho_ae_K_m, double *rho, double *ae, double *a, double *b, double *c):
    cdef:
        Py_ssize_t k
        double X, Y, Z #
        Py_ssize_t nz = nzg - 2* gw
    with nogil:
        for k in xrange(gw,nzg-gw):
            X = rho[k] * ae[k]/dt
            Y = rho_ae_K_m[k] * dzi * dzi
            Z = rho_ae_K_m[k-1] * dzi * dzi
            if k == gw:
                Z = 0.0
                Y = 0.0
            elif k == nzg-gw-1:
                Y = 0.0
            a[k-gw] = - Z/X
            b[k-gw] = 1.0 + Y/X + Z/X
            c[k-gw] = -Y/X

    return



cdef void tridiag_solve(Py_ssize_t nz, double *x, double *a, double *b, double *c):
    cdef:
        double * scratch = <double*> PyMem_Malloc(nz * sizeof(double))
        Py_ssize_t i
        double m

    scratch[0] = c[0]/b[0]
    x[0] = x[0]/b[0]

    with nogil:
        for i in xrange(1,nz):
            m = 1.0/(b[i] - a[i] * scratch[i-1])
            scratch[i] = c[i] * m
            x[i] = (x[i] - a[i] * x[i-1])*m


        for i in xrange(nz-2,-1,-1):
            x[i] = x[i] - scratch[i] * x[i+1]


    PyMem_Free(scratch)
    return








# Dustbin

cdef bint set_cloudbase_flag(double ql, bint current_flag) nogil:
    cdef bint new_flag
    if ql > 1.0e-8:
        new_flag = True
    else:
        new_flag = current_flag
    return  new_flag


