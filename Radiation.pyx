#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
from Grid cimport Grid
from ReferenceState cimport ReferenceState
from Variables cimport GridMeanVariables
from TimeStepping cimport TimeStepping
include 'parameters.pxi'
import numpy as np
cimport numpy as np
import netCDF4 as nc

cdef class RadiationBase:
    def __init__(self):
        return
    cpdef initialize(self, Grid Gr, ReferenceState Ref):
        return
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS):
        return
cdef class RadiationNone(RadiationBase):
    def __init__(self):
        RadiationBase.__init__(self)
        return
    cpdef initialize(self, Grid Gr, ReferenceState Ref):
        self.Gr = Gr
        self.Ref = Ref
        return
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS):
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

        # Initialize rrtmg_lw and rrtmg_sw
        cdef:
            double cpdair =np.float64(cpd)
        c_rrtmg_lw_init(&cpdair)
        c_rrtmg_sw_init(&cpdair)

        return
    cpdef initialize(self, Grid Gr, ReferenceState Ref):
        self.Gr = Gr
        self.Ref = Ref
        # Read in trace gas data
        lw_input_file = './RRTMG/lw/data/rrtmg_lw.nc'
        lw_gas = nc.Dataset(lw_input_file,  "r")

        lw_pressure = np.asarray(lw_gas.variables['Pressure'])
        lw_absorber = np.asarray(lw_gas.variables['AbsorberAmountMLS'])
        lw_absorber = np.where(lw_absorber>2.0, np.zeros_like(lw_absorber), lw_absorber)
        lw_ngas = lw_absorber.shape[1]
        lw_np = lw_absorber.shape[0]

        # 9 Gases: O3, CO2, CH4, N2O, O2, CFC11, CFC12, CFC22, CCL4
        # From rad_driver.f90, lines 546 to 552
        trace = np.zeros((9,lw_np),dtype=np.double,order='F')
        for i in xrange(lw_ngas):
            gas_name = ''.join(lw_gas.variables['AbsorberNames'][i,:])
            if 'O3' in gas_name:
                trace[0,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CO2' in gas_name:
                trace[1,:] = lw_absorber[:,i].reshape(1,lw_np)*self.co2_factor
            elif 'CH4' in gas_name:
                trace[2,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'N2O' in gas_name:
                trace[3,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'O2' in gas_name:
                trace[4,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CFC11' in gas_name:
                trace[5,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CFC12' in gas_name:
                trace[6,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CFC22' in gas_name:
                trace[7,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CCL4' in gas_name:
                trace[8,:] = lw_absorber[:,i].reshape(1,lw_np)

        # From rad_driver.f90, lines 585 to 620
        cdef:
            Py_ssize_t nz = self.Gr.nz
            double [:,:] trpath = np.zeros((nz + 1, 9),dtype=np.double,order='F')

        self.pi_full[0:nz] = self.Ref.p0[self.Gr.gw:self.Gr.nzg-self.Gr.gw]

        # plev = self.pi_full[:]/100.0
        for i in xrange(1, nz + self.n_ext + 1):
            trpath[i,:] = trpath[i-1,:]
            if (self.pi_full[i-1]/100.0 > lw_pressure[0]):
                trpath[i,:] = trpath[i,:] + (self.pi_full[i-1]/100.0 - np.max((self.pi_full[i]/100.0,lw_pressure[0])))/g*trace[:,0]
            for m in xrange(1,lw_np):
                plow = np.min((self.pi_full[i-1]/100.0,np.max((self.pi_full[i]/100.0, lw_pressure[m-1]))))
                pupp = np.min((self.pi_full[i-1]/100.0,np.max((self.pi_full[i]/100.0, lw_pressure[m]))))
                if (plow > pupp):
                    pmid = 0.5*(plow+pupp)
                    wgtlow = (pmid-lw_pressure[m])/(lw_pressure[m-1]-lw_pressure[m])
                    wgtupp = (lw_pressure[m-1]-pmid)/(lw_pressure[m-1]-lw_pressure[m])
                    trpath[i,:] = trpath[i,:] + (plow-pupp)/g*(wgtlow*trace[:,m-1]  + wgtupp*trace[:,m])
            if (self.pi_full[i]/100.0 < lw_pressure[lw_np-1]):
                trpath[i,:] = trpath[i,:] + (np.min((self.pi_full[i-1]/100.0,lw_pressure[lw_np-1]))-self.pi_full[i]/100.0)/g*trace[:,lw_np-1]

        tmpTrace = np.zeros((nz + self.n_ext,9),dtype=np.double,order='F')
        for i in xrange(9):
            for k in xrange(nz + self.n_ext):
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
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS):


        # Define input arrays for RRTM
        cdef:
            double [:,:] play_in = np.zeros((n_pencils,nz_full), dtype=np.double, order='F')
            double [:,:] plev_in = np.zeros((n_pencils,nz_full + 1), dtype=np.double, order='F')
            double [:,:] tlay_in = np.zeros((n_pencils,nz_full), dtype=np.double, order='F')
            double [:,:] tlev_in = np.zeros((n_pencils,nz_full + 1), dtype=np.double, order='F')
            double [:] tsfc_in = np.ones((n_pencils),dtype=np.double,order='F') * Sur.T_surface
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
            double [:,:] cldfr_in  = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] cicewp_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] cliqwp_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] reice_in  = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] reliq_in  = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:] coszen_in = np.ones((n_pencils),dtype=np.double,order='F') *self.coszen
            double [:] asdir_in = np.ones((n_pencils),dtype=np.double,order='F') * self.adir
            double [:] asdif_in = np.ones((n_pencils),dtype=np.double,order='F') * self.adif
            double [:] aldir_in = np.ones((n_pencils),dtype=np.double,order='F') * self.adir
            double [:] aldif_in = np.ones((n_pencils),dtype=np.double,order='F') * self.adif
            double [:,:,:] taucld_lw_in  = np.zeros((16,n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] tauaer_lw_in  = np.zeros((n_pencils,nz_full,16),dtype=np.double,order='F')
            double [:,:,:] taucld_sw_in  = np.zeros((14,n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] ssacld_sw_in  = np.zeros((14,n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] asmcld_sw_in  = np.zeros((14,n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] fsfcld_sw_in  = np.zeros((14,n_pencils,nz_full),dtype=np.double,order='F')
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

        return