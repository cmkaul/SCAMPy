from TimeStepping cimport TimeStepping
from Grid cimport Grid
cdef class NetCDFIO_Stats:
    cdef:
        Grid Gr
        object root_grp
        object profiles_grp
        object ts_grp

        str stats_file_name
        str stats_path
        str output_path
        str path_plus_file
        str uuid

        public double last_output_time
        public double frequency
        public bint do_output


    cpdef setup_stats_file(self)
    cpdef add_profile(self, var_name)
    cpdef add_reference_profile(self, var_name)
    cpdef add_ts(self, var_name)
    cpdef open_files(self)
    cpdef close_files(self)
    cpdef write_profile(self, var_name, double[:] data)
    cpdef write_reference_profile(self, var_name, double[:] data)
    cpdef write_ts(self, var_name, double data)
    cpdef write_simulation_time(self, double t)

