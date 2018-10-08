#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
from Grid cimport Grid
from ReferenceState cimport ReferenceState
from Variables cimport GridMeanVariables
from TimeStepping cimport TimeStepping
from EDMF_Updrafts cimport UpdraftVariables
from EDMF_Environment cimport EnvironmentVariables
from NetCDFIO cimport NetCDFIO_Stats
include 'parameters.pxi'
import numpy as np
cimport numpy as np
import netCDF4 as nc
from libc.math cimport fmax, fmin
from utility_functions cimport interp2pt
from forcing_functions cimport  convert_forcing_entropy, convert_forcing_thetal
from thermodynamic_functions cimport cpm_c
import pylab as plt
cdef class RadiationBase:
    def __init__(self):
        return
    cpdef initialize(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV):
        self.Gr = Gr
        self.Ref = Ref
        self.gm_T_tendency = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
        self.gm_heating_rate_lw = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
        self.gm_heating_rate_sw = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
        self.gm_uflux_lw = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
        self.gm_dflux_lw = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
        self.gm_uflux_sw = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
        self.gm_dflux_sw = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')


        if GMV.H.name == 's':
            self.convert_forcing_prog_fp = convert_forcing_entropy
        elif GMV.H.name == 'thetal':
            self.convert_forcing_prog_fp = convert_forcing_thetal

        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats ):
        Stats.add_profile('rad_T_tendency')
        Stats.add_profile('rad_heating_lw')
        Stats.add_profile('rad_heating_sw')
        # Stats.add_profile('rad_uflux_lw')
        # Stats.add_profile('rad_dflux_lw')
        # Stats.add_profile('rad_uflux_sw')
        # Stats.add_profile('rad_dflux_sw')
        Stats.add_ts('rad_srf_lw_up')
        Stats.add_ts('rad_srf_lw_down')
        Stats.add_ts('rad_srf_sw_up')
        Stats.add_ts('rad_srf_sw_down')
        Stats.add_ts('rad_toa_lw_up')
        Stats.add_ts('rad_toa_sw_up')
        Stats.add_ts('rad_toa_sw_down')
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        cdef:
            Py_ssize_t kmin = self.Gr.gw
            Py_ssize_t kmax = self.Gr.nzg-self.Gr.gw
        Stats.write_profile('rad_T_tendency', self.gm_T_tendency[kmin:kmax])
        Stats.write_profile('rad_heating_lw', self.gm_heating_rate_lw[kmin:kmax])
        Stats.write_profile('rad_heating_sw', self.gm_heating_rate_sw[kmin:kmax])
        Stats.write_ts('rad_srf_lw_up', self.srf_lw_up)
        Stats.write_ts('rad_srf_lw_down', self.srf_lw_down)
        Stats.write_ts('rad_srf_sw_up', self.srf_sw_up)
        Stats.write_ts('rad_srf_sw_down', self.srf_sw_down)
        Stats.write_ts('rad_toa_lw_up', self.toa_lw_up)
        Stats.write_ts('rad_toa_sw_up', self.toa_sw_up)
        Stats.write_ts('rad_toa_sw_down', self.toa_sw_down)
        return
    cpdef update(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                 EnvironmentVariables EnvVar, TimeStepping TS, double Tsurface):
        cdef:
            Py_ssize_t k

        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            # Apply large-scale horizontal advection tendencies
            qv = GMV.QT.values[k] - GMV.QL.values[k]
            GMV.H.tendencies[k] += self.convert_forcing_prog_fp(self.Ref.p0_half[k],GMV.QT.values[k],
                                                                qv, GMV.T.values[k], 0.0, self.gm_T_tendency[k])


        # set surface values
        self.srf_lw_up = self.gm_uflux_lw[self.Gr.gw-1]
        self.srf_lw_down = self.gm_dflux_lw[self.Gr.gw-1]
        self.srf_sw_up = self.gm_uflux_sw[self.Gr.gw-1]
        self.srf_sw_down = self.gm_dflux_sw[self.Gr.gw-1]

        self.toa_lw_up = self.gm_uflux_lw[self.Gr.gw-1+self.Gr.nz]
        self.toa_sw_up = self.gm_uflux_sw[self.Gr.gw-1+self.Gr.nz]
        self.toa_sw_down = self.gm_dflux_sw[self.Gr.gw-1+self.Gr.nz]


        return
cdef class RadiationNone(RadiationBase):
    def __init__(self):
        RadiationBase.__init__(self)
        return
    cpdef initialize(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV):
        RadiationBase.initialize(self, Gr, Ref, GMV)
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        return
    cpdef update(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                 EnvironmentVariables EnvVar, TimeStepping TS, double Tsurface):
        return
# Note: the RRTM modules are compiled in the 'RRTMG' directory:
cdef extern:
    void c_rrtmg_lw_init(double *cpdair)
    void c_rrtmg_lw (
             int *ncol    ,int *nlay    ,int *icld    ,int *idrv    ,
             double *play    ,double *plev    ,double *tlay    ,double *tlev    ,double *tsfc    ,
             double *h2ovmr  ,double *o3vmr   ,double *co2vmr  ,double *ch4vmr  ,double *n2ovmr  ,double *o2vmr,
             double *cfc11vmr,double *cfc12vmr,double *cfc22vmr,double *ccl4vmr ,double *emis    ,
             int *inflglw ,int *iceflglw,int *liqflglw,double *cldfr   ,
             double *taucld  ,double *cicewp  ,double *cliqwp  ,double *reice   ,double *reliq   ,
             double *tauaer  ,
             double *uflx    ,double *dflx    ,double *hr      ,double *uflxc   ,double *dflxc,  double *hrc,
             double *duflx_dt,double *duflxc_dt )
    void c_rrtmg_sw_init(double *cpdair)
    void c_rrtmg_sw (int *ncol    ,int *nlay    ,int *icld    ,int *iaer    ,
             double *play    ,double *plev    ,double *tlay    ,double *tlev    ,double *tsfc    ,
             double *h2ovmr  ,double *o3vmr   ,double *co2vmr  ,double *ch4vmr  ,double *n2ovmr  ,double *o2vmr,
             double *asdir   ,double *asdif   ,double *aldir   ,double *aldif   ,
             double *coszen  ,double *adjes   ,int *dyofyr  ,double *scon    ,
             int *inflgsw ,int *iceflgsw,int *liqflgsw,double *cldfr   ,
             double *taucld  ,double *ssacld  ,double *asmcld  ,double *fsfcld  ,
             double *cicewp  ,double *cliqwp  ,double *reice   ,double *reliq   ,
             double *tauaer  ,double *ssaaer  ,double *asmaer  ,double *ecaer   ,
             double *swuflx  ,double *swdflx  ,double *swhr    ,double *swuflxc ,double *swdflxc ,double *swhrc)


cdef class RadiationRRTM(RadiationBase):
    def __init__(self, namelist):
        # RadiationBase.__init__(self)
        try:
            self.co2_factor = namelist['radiation']['co2_factor']
        except:
            self.co2_factor = 1.0
        try:
            self.dyofyr = namelist['radiation']['day_of_year']
        except:
            self.dyofyr = 0
        try:
            self.adjes = namelist['radiation']['adjes']
        except:
            self.adjes = 1.0
        try:
            self.scon = namelist['radiation']['solar_constant']
        except:
            self.scon = 1365.0
        try:
            self.coszen = namelist['radiation']['coszen']
        except:
            self.coszen = 0.354
        try:
            self.adif = namelist['radiation']['adif']
        except:
            self.adif = 0.1
        try:
            self.adir = namelist['radiation']['adir']
        except:
            self.adir =  (.026/(self.coszen**1.7 + .065)+(.15*(self.coszen-0.10)*(self.coszen-0.50)*(self.coszen- 1.00)))
            # for defaults  this gives adir = 0.11

        try:
            self.compute_on_subdomains = namelist['radiation']['compute_on_subdomains']
        except:
            self.compute_on_subdomains = False

        # Initialize rrtmg_lw and rrtmg_sw
        cdef:
            double cpdair =np.float64(cpd)
        c_rrtmg_lw_init(&cpdair)
        c_rrtmg_sw_init(&cpdair)

        return
    cpdef initialize(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV):
        RadiationBase.initialize(self, Gr, Ref, GMV)
        # Read in trace gas data
        lw_input_file = './RRTMG/lw/data/rrtmg_lw.nc'
        lw_gas = nc.Dataset(lw_input_file,  "r")
        lw_pressure = np.asarray(lw_gas.variables['Pressure'])
        lw_absorber = np.asarray(lw_gas.variables['AbsorberAmountMLS'])
        lw_absorber = np.where(lw_absorber>2.0, np.zeros_like(lw_absorber), lw_absorber)
        lw_ngas = lw_absorber.shape[1]
        lw_np = lw_absorber.shape[0]

        trace = np.zeros((9,lw_np),dtype=np.double,order='F')

        # 9 Gases: O3, CO2, CH4, N2O, O2, CFC11, CFC12, CFC22, CCL4
        # From rad_driver.f90, lines 546 to 552

        for i in xrange(lw_ngas):
            gas_name = ''.join(lw_gas.variables['AbsorberNames'][i,:])
            if 'O3' in gas_name:
                trace[0,:] = np.reshape(lw_absorber[:,i],(1,lw_np))
            elif 'CO2' in gas_name:
                trace[1,:] = np.reshape(lw_absorber[:,i],(1,lw_np))*self.co2_factor
            elif 'CH4' in gas_name:
                trace[2,:] = np.reshape(lw_absorber[:,i],(1,lw_np))
            elif 'N2O' in gas_name:
                trace[3,:] = np.reshape(lw_absorber[:,i],(1,lw_np))
            elif 'O2' in gas_name:
                trace[4,:] = np.reshape(lw_absorber[:,i],(1,lw_np))
            elif 'CFC11' in gas_name:
                trace[5,:] = np.reshape(lw_absorber[:,i],(1,lw_np))
            elif 'CFC12' in gas_name:
                trace[6,:] = np.reshape(lw_absorber[:,i],(1,lw_np))
            elif 'CFC22' in gas_name:
                trace[7,:] = np.reshape(lw_absorber[:,i],(1,lw_np))
            elif 'CCL4' in gas_name:
                trace[8,:] = np.reshape(lw_absorber[:,i],(1,lw_np))

        # From rad_driver.f90, lines 585 to 620
        cdef:
            Py_ssize_t nz = self.Gr.nz
        trpath = np.zeros((nz + 1, 9),dtype=np.double,order='F')
        self.pi_full   = np.zeros((nz + 1),dtype=np.double,order='c')
        self.p_full = np.zeros((nz ),dtype=np.double,order='c')

        self.pi_full[0:nz+1] = self.Ref.p0[self.Gr.gw:self.Gr.nzg-self.Gr.gw+1]
        self.p_full[0:nz] = self.Ref.p0_half[self.Gr.gw:self.Gr.nzg-self.Gr.gw]

        # plev = self.pi_full[:]/100.0
        for i in xrange(1, nz + 1):
            trpath[i,:] = trpath[i-1,:]
            if (self.pi_full[i-1]/100.0 > lw_pressure[0]):
                trpath[i,:] = trpath[i,:] + (self.pi_full[i-1]/100.0 - np.max((self.pi_full[i]/100.0,lw_pressure[0])))/g*trace[:,0]
            for m in xrange(1,lw_np):
                plow = np.min((self.pi_full[i-1]/100.0,np.max((self.pi_full[i]/100.0, lw_pressure[m-1]))))
                pupp = np.min((self.pi_full[i-1]/100.0,np.max((self.pi_full[i]/100.0, lw_pressure[m]))))
                if plow > pupp:
                    pmid = 0.5*(plow+pupp)
                    wgtlow = (pmid-lw_pressure[m])/(lw_pressure[m-1]-lw_pressure[m])
                    wgtupp = (lw_pressure[m-1]-pmid)/(lw_pressure[m-1]-lw_pressure[m])
                    trpath[i,:] = trpath[i,:] + (plow-pupp)/g*(wgtlow*trace[:,m-1]  + wgtupp*trace[:,m])
            if (self.pi_full[i]/100.0 < lw_pressure[lw_np-1]):
                trpath[i,:] = trpath[i,:] + (np.min((self.pi_full[i-1]/100.0,lw_pressure[lw_np-1]))-self.pi_full[i]/100.0)/g*trace[:,lw_np-1]

        tmpTrace = np.zeros((nz ,9),dtype=np.double,order='F')
        for i in xrange(9):
            for k in xrange(nz ):
                tmpTrace[k,i] = g*100.0/(self.pi_full[k]-self.pi_full[k+1])*(trpath[k+1,i]-trpath[k,i])


        self.o3vmr  = np.array(tmpTrace[:,0],dtype=np.double, order='F')
        self.co2vmr = np.array(tmpTrace[:,1],dtype=np.double, order='F')
        self.ch4vmr =  np.array(tmpTrace[:,2],dtype=np.double, order='F')
        self.n2ovmr =  np.array(tmpTrace[:,3],dtype=np.double, order='F')
        self.o2vmr  =  np.array(tmpTrace[:,4],dtype=np.double, order='F')
        self.cfc11vmr =  np.array(tmpTrace[:,5],dtype=np.double, order='F')
        self.cfc12vmr =  np.array(tmpTrace[:,6],dtype=np.double, order='F')
        self.cfc22vmr = np.array( tmpTrace[:,7],dtype=np.double, order='F')
        self.ccl4vmr  =  np.array(tmpTrace[:,8],dtype=np.double, order='F')



        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        RadiationBase.initialize_io(self,Stats)
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        RadiationBase.io(self,Stats)
        return
    cpdef update(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                 EnvironmentVariables EnvVar, TimeStepping TS, double Tsurface):


        # Define input arrays for RRTM
        cdef:
            Py_ssize_t i, k, krad
            Py_ssize_t nz_full = self.Gr.nz
            Py_ssize_t n_pencils = 1 #2 + UpdVar.n_updrafts
            Py_ssize_t nbndlw = 16
            Py_ssize_t ngptlw = 140
            Py_ssize_t ngptsw = 112
            double [:,:] play_in = np.zeros((n_pencils,nz_full), dtype=np.double, order='F')
            double [:,:] plev_in = np.zeros((n_pencils,nz_full + 1), dtype=np.double, order='F')
            double [:,:] tlay_in = np.zeros((n_pencils,nz_full), dtype=np.double, order='F')
            double [:,:] tlev_in = np.zeros((n_pencils,nz_full + 1), dtype=np.double, order='F')
            double [:] tsfc_in = np.ones((n_pencils),dtype=np.double,order='F') * Tsurface
            double [:,:] h2ovmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] o3vmr_in  = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] co2vmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] ch4vmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] n2ovmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] o2vmr_in  = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] cfc11vmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] cfc12vmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] cfc22vmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] ccl4vmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] emis_in = np.ones((n_pencils,16),dtype=np.double,order='F') * 0.95
            # changed for McICA
            double [:,:,:] cldfr_lw_in  = np.zeros((ngptlw, n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] cicewp_lw_in = np.zeros((ngptlw, n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] cliqwp_lw_in = np.zeros((ngptlw, n_pencils,nz_full),dtype=np.double,order='F')
            # changed for McICA
            double [:,:,:] cldfr_sw_in  = np.zeros((ngptsw, n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] cicewp_sw_in = np.zeros((ngptsw, n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] cliqwp_sw_in = np.zeros((ngptsw, n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] reice_in  = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] reliq_in  = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:] coszen_in = np.ones((n_pencils),dtype=np.double,order='F') *self.coszen
            double [:] asdir_in = np.ones((n_pencils),dtype=np.double,order='F') * self.adir
            double [:] asdif_in = np.ones((n_pencils),dtype=np.double,order='F') * self.adif
            double [:] aldir_in = np.ones((n_pencils),dtype=np.double,order='F') * self.adir
            double [:] aldif_in = np.ones((n_pencils),dtype=np.double,order='F') * self.adif
            #changed for McICA
            double [:,:,:] taucld_lw_in  = np.zeros((ngptlw,n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] tauaer_lw_in  = np.zeros((n_pencils,nz_full,nbndlw),dtype=np.double,order='F')
            #changed for McICA
            double [:,:,:] taucld_sw_in  = np.zeros((ngptsw,n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] ssacld_sw_in  = np.zeros((ngptsw,n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] asmcld_sw_in  = np.zeros((ngptsw,n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] fsfcld_sw_in  = np.zeros((ngptsw,n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] tauaer_sw_in  = np.zeros((n_pencils,nz_full,14),dtype=np.double,order='F')
            double [:,:,:] ssaaer_sw_in  = np.zeros((n_pencils,nz_full,14),dtype=np.double,order='F')
            double [:,:,:] asmaer_sw_in  = np.zeros((n_pencils,nz_full,14),dtype=np.double,order='F')
            double [:,:,:] ecaer_sw_in  = np.zeros((n_pencils,nz_full,6),dtype=np.double,order='F')

            # Output
            double[:,:] uflx_lw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] dflx_lw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] hr_lw_out = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double[:,:] uflxc_lw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] dflxc_lw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] hrc_lw_out = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double[:,:] duflx_dt_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] duflxc_dt_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] uflx_sw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] dflx_sw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] hr_sw_out = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double[:,:] uflxc_sw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] dflxc_sw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] hrc_sw_out = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')

            double rv_to_reff = np.exp(np.log(1.2)**2.0)*10.0*1000.0
            double gmv_qv, env_qv, upd_qv, gmv_rl, env_rl, upd_rl

            double [:,:] qv = np.zeros((n_pencils, nz_full),dtype=np.double)


        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            krad = k - self.Gr.gw

            play_in[0,krad] = self.p_full[krad]/100.0
            # play_in[1,krad] = self.p_full[krad]/100.0

            tlay_in[0,krad] = GMV.T.values[k]
            # tlay_in[1,krad] = EnvVar.T.values[k]

            gmv_qv = GMV.QT.values[k] - GMV.QL.values[k]
            qv[0,k] = gmv_qv
            gmv_rl =  GMV.QL.values[k]/ (1.0 - gmv_qv)
            h2ovmr_in[0,krad] = gmv_qv/ (1.0 - gmv_qv)* Rv/Rd
            env_qv = EnvVar.QT.values[k] - EnvVar.QL.values[k]
            # qv[1,k] = env_qv
            env_rl = EnvVar.QL.values[k]/ (1.0 - env_qv)
            # h2ovmr_in[1,krad] = env_qv/ (1.0 - env_qv)* Rv/Rd

            cliqwp_lw_in[:,0,krad] =  (GMV.QL.values[k]/ (1.0 - gmv_qv)*1.0e3*(self.pi_full[krad] - self.pi_full[krad+1])/g)
            cliqwp_sw_in[:,0,krad] =  cliqwp_lw_in[0,0,krad]
            # cliqwp_in[1,krad] = (EnvVar.QL.values[k]/ (1.0 - env_qv)*1.0e3*(self.pi_full[krad] - self.pi_full[krad+1])/g)
            # cicewp_in[0,krad] = (GMV.QI.values[k]/ (1.0 - gmv_qv)*1.0e3*(self.pi_full[krad] - self.pi_full[krad+1])/g)
            # cicewp_in[1,krad] = (EnvVar.QI.values[k]/ (1.0 - env_qv)*1.0e3*(self.pi_full[krad] - self.pi_full[krad+1])/g)

        
            cldfr_lw_in[:,0,krad] = (1.0-UpdVar.Area.bulkvalues[k]) * EnvVar.CF.values[k]
            cldfr_sw_in[:,0,krad] = cldfr_lw_in[0,0,krad]
            # cldfr_in[1,krad] = EnvVar.CF.values[k]

            reliq_in[0, krad] = ((3.0*self.p_full[k]/Rd/tlay_in[0,krad]*gmv_rl/fmax(cldfr_lw_in[0,0,krad],1.0e-6))/(4.0*pi*1.0e3*100.0))**(1.0/3.0)
            reliq_in[0, krad] = fmin(fmax(reliq_in[0, krad] * rv_to_reff, 2.5), 60.0)
            # reliq_in[1, krad] = ((3.0*self.p_full[k]/Rd/tlay_in[1,krad]*env_rl/fmax(cldfr_in[1,krad],1.0e-6))/(4.0*pi*1.0e3*100.0))**(1.0/3.0)
            # reliq_in[1, krad] = fmin(fmax(reliq_in[1, krad] * rv_to_reff, 2.5), 60.0)
            #
            o3vmr_in[0, krad] = self.o3vmr[krad]
            co2vmr_in[0, krad] = self.co2vmr[krad]
            ch4vmr_in[0, krad] = self.ch4vmr[krad]
            n2ovmr_in[0, krad] = self.n2ovmr[krad]
            o2vmr_in [0, krad] = self.o2vmr[krad]
            cfc11vmr_in[0, krad] = self.cfc11vmr[krad]
            cfc12vmr_in[0, krad] = self.cfc12vmr[krad]
            cfc22vmr_in[0, krad] = self.cfc22vmr[krad]
            ccl4vmr_in[0, krad] = self.ccl4vmr[krad]

            # o3vmr_in[1, krad] = self.o3vmr[krad]
            # co2vmr_in[1, krad] = self.co2vmr[krad]
            # ch4vmr_in[1, krad] = self.ch4vmr[krad]
            # n2ovmr_in[1, krad] = self.n2ovmr[krad]
            # o2vmr_in [1, krad] = self.o2vmr[krad]
            # cfc11vmr_in[1, krad] = self.cfc11vmr[krad]
            # cfc12vmr_in[1, krad] = self.cfc12vmr[krad]
            # cfc22vmr_in[1, krad] = self.cfc22vmr[krad]
            # ccl4vmr_in[1, krad] = self.ccl4vmr[krad]

            #
            # for i in xrange(UpdVar.n_updrafts):
            #     play_in[2+i,krad] = self.p_full[krad]/100.0
            #     tlay_in[2+i, krad] = UpdVar.T.values[i, k]
            #     upd_qv = UpdVar.QT.values[i,k] - UpdVar.QL.values[i,k]
            #     qv[2+i,k] = upd_qv
            #     upd_rl = UpdVar.QL.values[i,k]/ (1.0 - upd_qv)
            #     h2ovmr_in[2+i,krad] = upd_qv/ (1.0 - upd_qv)* Rv/Rd
            #     cliqwp_in[2+i,krad] = (UpdVar.QL.values[i,k]/ (1.0 - upd_qv)*1.0e3*(self.pi_full[krad] - self.pi_full[krad+1])/g)
            #     # cicewp_in[2+i,krad] = (UpdVar.QI.values[i,k]/ (1.0 - upd_qv)*1.0e3*(self.pi_full[krad] - self.pi_full[krad+1])/g)
            #     cldfr_in[2+i, krad] = np.ceil(UpdVar.QL.values[i,k])
            #     cldfr_in[0,krad] += UpdVar.Area.values[i,k] * cldfr_in[2+i, krad]
            #
            #     reliq_in[2+i, krad] = ((3.0*self.p_full[k]/Rd/tlay_in[2+i,krad]*upd_rl/fmax(cldfr_in[2+i,krad],1.0e-6))/(4.0*pi*1.0e3*100.0))**(1.0/3.0)
            #     reliq_in[2+i, krad] = fmin(fmax(reliq_in[2+i, krad] * rv_to_reff, 2.5), 60.0)
            #
            #     o3vmr_in[2+i, krad] = self.o3vmr[krad]
            #     co2vmr_in[2+i, krad] = self.co2vmr[krad]
            #     ch4vmr_in[2+i, krad] = self.ch4vmr[krad]
            #     n2ovmr_in[2+i, krad] = self.n2ovmr[krad]
            #     o2vmr_in [2+i, krad] = self.o2vmr[krad]
            #     cfc11vmr_in[2+i, krad] = self.cfc11vmr[krad]
            #     cfc12vmr_in[2+i, krad] = self.cfc12vmr[krad]
            #     cfc22vmr_in[2+i, krad] = self.cfc22vmr[krad]
            #     ccl4vmr_in[2+i, krad] = self.ccl4vmr[krad]

        # for i in xrange(2+UpdVar.n_updrafts):
        for i in xrange(1):
            tlev_in[i, 0] = Tsurface
            plev_in[i,0] = self.pi_full[0]/100.0
            plev_in[i, nz_full] =self.pi_full[nz_full]/100.0
            for krad in xrange(1,nz_full):
                tlev_in[i,krad] = interp2pt(tlay_in[i,krad-1], tlay_in[i,krad])
                plev_in[i,krad] = self.pi_full[krad]/100.0
            tlev_in[i, nz_full] = 2.0*tlay_in[i,nz_full-1] - tlev_in[i,nz_full-1]
            plev_in[i,nz_full] = self.pi_full[nz_full]/100.0

        # plt.figure('Tlev, tlay')
        # plt.plot(tlev_in[0,:], plev_in[0,:], '-b')
        # plt.plot(tlay_in[0,:], play_in[0,:], '--r')
        # plt.gca().invert_yaxis()
        #
        # plt.figure('p vs z')
        # plt.plot(self.Ref.p0[self.Gr.gw-1:self.Gr.nzg-self.Gr.gw],self.Gr.z[self.Gr.gw-1:self.Gr.nzg-self.Gr.gw],  '-b')
        # plt.plot( self.Ref.p0_half[self.Gr.gw:self.Gr.nzg-self.Gr.gw], self.Gr.z_half[self.Gr.gw:self.Gr.nzg-self.Gr.gw],'--r')
        #



        cdef:
            int ncol = 1
            int nlay = nz_full
            int icld = 1
            int idrv = 0
            int iaer = 0
            int inflglw = 2
            int iceflglw = 3
            int liqflglw = 1
            int inflgsw = 2
            int iceflgsw = 3
            int liqflgsw = 1

        c_rrtmg_lw (
             &ncol    ,&nlay    ,&icld    ,&idrv,
             &play_in[0,0]    ,&plev_in[0,0]    ,&tlay_in[0,0]    ,&tlev_in[0,0]    ,&tsfc_in[0]    ,
             &h2ovmr_in[0,0]  ,&o3vmr_in[0,0]   ,&co2vmr_in[0,0]  ,&ch4vmr_in[0,0]  ,&n2ovmr_in[0,0]  ,&o2vmr_in[0,0],
             &cfc11vmr_in[0,0],&cfc12vmr_in[0,0],&cfc22vmr_in[0,0],&ccl4vmr_in[0,0] ,&emis_in[0,0]    ,
             &inflglw ,&iceflglw,&liqflglw,&cldfr_lw_in[0,0,0]   ,
             &taucld_lw_in[0,0,0]  ,&cicewp_lw_in[0,0,0]  ,&cliqwp_lw_in[0,0,0]  ,&reice_in[0,0]   ,&reliq_in[0,0]   ,
             &tauaer_lw_in[0,0,0]  ,
             &uflx_lw_out[0,0]    ,&dflx_lw_out[0,0]    ,&hr_lw_out[0,0]      ,&uflxc_lw_out[0,0]   ,&dflxc_lw_out[0,0],  &hrc_lw_out[0,0],
             &duflx_dt_out[0,0],&duflxc_dt_out[0,0] )


        c_rrtmg_sw (
            &ncol, &nlay, &icld, &iaer, &play_in[0,0], &plev_in[0,0], &tlay_in[0,0], &tlev_in[0,0],&tsfc_in[0],
            &h2ovmr_in[0,0], &o3vmr_in[0,0], &co2vmr_in[0,0], &ch4vmr_in[0,0], &n2ovmr_in[0,0],&o2vmr_in[0,0],
             &asdir_in[0]   ,&asdif_in[0]   ,&aldir_in[0]   ,&aldif_in[0]   ,
             &coszen_in[0]  ,&self.adjes   ,&self.dyofyr  ,&self.scon   ,
             &inflgsw ,&iceflgsw,&liqflgsw,&cldfr_sw_in[0,0,0]   ,
             &taucld_sw_in[0,0,0]  ,&ssacld_sw_in[0,0,0]  ,&asmcld_sw_in[0,0,0]  ,&fsfcld_sw_in[0,0,0]  ,
             &cicewp_sw_in[0,0,0]  ,&cliqwp_sw_in[0,0,0]  ,&reice_in[0,0]   ,&reliq_in[0,0]   ,
             &tauaer_sw_in[0,0,0]  ,&ssaaer_sw_in[0,0,0]  ,&asmaer_sw_in[0,0,0]  ,&ecaer_sw_in[0,0,0]   ,
             &uflx_sw_out[0,0]    ,&dflx_sw_out[0,0]    ,&hr_sw_out[0,0]      ,&uflxc_sw_out[0,0]   ,&dflxc_sw_out[0,0], &hrc_sw_out[0,0])



        if self.compute_on_subdomains:
            self.zero_fluxes()
            # for i in xrange(1,n_pencils):
            #     for k in xrange(nz_full):
            #        kgrid = k + self.Gr.gw
            #        self.gm_T_tendency[kgrid] += (hr_lw_out[i,k] + hr_sw_out[i,k])/86400.0
            #        self.heating_rate_lw[kgrid] += (hr_lw_out[i,k]) * self.Ref.rho0_half[kgrid] * cpm_c(qv[i,kgrid])/86400.0
            #        self.heating_rate_sw[kgrid] += (hr_sw_out[i,k]) * self.Ref.rho0_half[kgrid] * cpm_c(qv[i,kgrid])/86400.0
            #     for k in xrange(nz_full+1):
            #         kgrid = k + self.Gr.gw - 1
            #         self.gm_uflux_lw[kgrid] += uflx_lw_out[i,k]
            #         self.gm_dflux_lw[kgrid] += dflx_lw_out[i,k]
            #         self.gm_uflux_sw[kgrid] += uflx_sw_out[i,k]
            #         self.gm_dflux_sw[kgrid] += dflx_sw_out[i,k]
        else:
            for k in xrange(nz_full):
                kgrid = k + self.Gr.gw
                self.gm_T_tendency[kgrid] = (hr_lw_out[0,k] + hr_sw_out[0,k])/86400.0
                self.gm_heating_rate_lw[kgrid] = (hr_lw_out[0,k]) * self.Ref.rho0_half[kgrid] * cpm_c(qv[0,kgrid])/86400.0
                self.gm_heating_rate_sw[kgrid] = (hr_sw_out[0,k]) * self.Ref.rho0_half[kgrid] * cpm_c(qv[0,kgrid])/86400.0
            for k in xrange(nz_full+1):
                kgrid = k + self.Gr.gw - 1
                self.gm_uflux_lw[kgrid] = uflx_lw_out[0,k]
                self.gm_dflux_lw[kgrid] = dflx_lw_out[0,k]
                self.gm_uflux_sw[kgrid] = uflx_sw_out[0,k]
                self.gm_dflux_sw[kgrid] = dflx_sw_out[0,k]

        RadiationBase.update(self, GMV, UpdVar, EnvVar, TS, Tsurface)
        # plt.figure('rad_T_tendemcy')
        # plt.plot(np.divide(np.add(hr_lw_out[0,:],hr_sw_out[0,:]),86400.0), play_in[0,:],'-o')
        #
        # plt.gca().invert_yaxis()
        # plt.show()


        return

    cpdef zero_fluxes(self):
        for k in xrange(self.Gr.nzg):
            self.gm_T_tendency[k] = 0.0
            self.gm_heating_rate_lw[k] = 0.0
            self.gm_heating_rate_sw[k] = 0.0
            self.gm_uflux_lw[k] = 0.0
            self.gm_dflux_lw[k] = 0.0
            self.gm_uflux_sw[k] = 0.0
            self.gm_dflux_sw[k] = 0.0
        return

