from NetCDFIO cimport NetCDFIO_Stats
from Grid cimport  Grid
from ReferenceState cimport ReferenceState
from Variables cimport VariableDiagnostic, GridMeanVariables
#from TimeStepping cimport  TimeStepping

cdef class EnvironmentVariable:
    cdef:
        double [:] values
        double [:] flux
        str loc
        str kind
        str name
        str units

cdef class EnvironmentVariables:
    cdef:

        EnvironmentVariable W
        EnvironmentVariable QT
        EnvironmentVariable QL
        EnvironmentVariable QR
        EnvironmentVariable H
        EnvironmentVariable THL
        EnvironmentVariable T
        EnvironmentVariable B
        EnvironmentVariable TKE
        EnvironmentVariable Hvar
        EnvironmentVariable QTvar
        EnvironmentVariable HQTcov
        EnvironmentVariable CF
        EnvironmentVariable THVvar
        Grid Gr
        bint use_tke
        bint use_scalar_var
        bint use_prescribed_scalar_var
        double prescribed_QTvar
        double prescribed_Hvar
        double prescribed_HQTcov
        bint use_sommeria_deardorff
        bint use_quadrature
        str EnvThermo_scheme

    cpdef initialize_io(self, NetCDFIO_Stats Stats )
    cpdef io(self, NetCDFIO_Stats Stats)

cdef class EnvironmentThermodynamics:
    cdef:
        Grid Gr
        ReferenceState Ref
        Py_ssize_t quadrature_order

        double (*t_to_prog_fp)(double p0, double T,  double qt, double ql, double qi)   nogil
        double (*prog_to_t_fp)(double H, double pd, double pv, double qt ) nogil

        double [:] qt_dry
        double [:] th_dry
        double [:] t_cloudy
        double [:] qv_cloudy
        double [:] qt_cloudy
        double [:] th_cloudy

        double [:] Hvar_rain_dt
        double [:] QTvar_rain_dt
        double [:] HQTcov_rain_dt

        double max_supersaturation

        void update_EnvVar(self,    long k, EnvironmentVariables EnvVar, double T, double H, double qt, double ql, double qr, double alpha) nogil
        void update_cloud_dry(self, long k, EnvironmentVariables EnvVar, double T, double H, double qt, double ql, double qv) nogil

        void eos_update_SA_mean(self, EnvironmentVariables EnvVar, bint in_Env)
        void eos_update_SA_sgs(self, EnvironmentVariables EnvVar, bint in_Env)#, TimeStepping TS)
        void sommeria_deardorff(self, EnvironmentVariables EnvVar)

    cpdef satadjust(self, EnvironmentVariables EnvVar, bint in_Env)#, TimeStepping TS)
