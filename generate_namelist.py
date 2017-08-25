import argparse
import json
import pprint
from sys import exit
import uuid
import ast


def main():
    parser = argparse.ArgumentParser(prog='Namelist Generator')
    parser.add_argument('case_name')

    args = parser.parse_args()

    case_name = args.case_name

    if case_name == 'Bomex':
        namelist = Bomex()
    elif case_name == 'Bomex_pulse':
        namelist = Bomex_pulse()
    elif case_name == 'Bomex_pulses':
        namelist = Bomex_pulses()
    elif case_name == 'Bomex_cosine':
        namelist = Bomex_cosine()
    elif case_name == 'Soares':
        namelist = Soares()
    elif case_name == 'Rico':
        namelist = Rico()
    elif case_name == 'TRMM_LBA':
        namelist = TRMM_LBA()
    elif case_name == 'ARM_SGP':
        namelist = ARM_SGP()
    elif case_name == 'SCMS':
        namelist = SCMS()
    elif case_name == 'GATE_III':
        namelist = GATE_III()
    else:
        print('Not a vaild case name')
        exit()

    write_file(namelist)


def Soares():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 150
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 20.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['thermal_variable'] = 'thetal'


    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 60.0
    namelist['time_stepping']['t_max'] = 8 * 3600.0


    namelist['turbulence'] = {}
    namelist['turbulence']['scheme'] = 'EDMF_BulkSteady'
    namelist['turbulence']['EDMF_BulkSteady'] = {}
    namelist['turbulence']['EDMF_BulkSteady']['updraft_number'] = 1
    namelist['turbulence']['EDMF_BulkSteady']['constant_area'] = True
    namelist['turbulence']['EDMF_BulkSteady']['surface_area'] = 0.1
    namelist['turbulence']['EDMF_BulkSteady']['entrainment'] = 'dry'


    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['frequency'] = 60.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Soares'
    namelist['meta']['casename'] = 'Soares'


    return namelist

def Bomex():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 75
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 100 / 2.5


    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['thermal_variable'] = 'thetal'

    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 20.0
    namelist['time_stepping']['t_max'] = 21600.0


    namelist['turbulence'] = {}
    namelist['turbulence']['scheme'] = 'EDMF_PrognosticTKE'
    namelist['turbulence']['EDMF_PrognosticTKE'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number'] = 1
    namelist['turbulence']['EDMF_PrognosticTKE']['constant_area'] = False
    namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'] = 'b_w2'
    namelist['turbulence']['EDMF_PrognosticTKE']['use_local_micro'] = True
    namelist['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity'] = False
    namelist['turbulence']['EDMF_PrognosticTKE']['updraft_surface_height'] = 0.0
    namelist['turbulence']['EDMF_PrognosticTKE']['extrapolate_buoyancy'] = True

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['frequency'] = 60.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Bomex'
    namelist['meta']['casename'] = 'Bomex'


    return namelist

def Bomex_pulse():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 75
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 100 / 2.5


    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['thermal_variable'] = 'thetal'

    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 1.0
    namelist['time_stepping']['t_max'] = 3*3600.0


    namelist['turbulence'] = {}
    namelist['turbulence']['scheme'] = 'EDMF_PrognosticTKE'
    namelist['turbulence']['EDMF_PrognosticTKE'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number'] = 1
    namelist['turbulence']['EDMF_PrognosticTKE']['constant_area'] = False
    namelist['turbulence']['EDMF_PrognosticTKE']['surface_area'] = 0.1


    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['frequency'] = 1.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Bomex_pulse'
    namelist['meta']['casename'] = 'Bomex_pulse'


    return namelist

def Bomex_pulses():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 75
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 100 / 2.5


    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['thermal_variable'] = 'thetal'

    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 1.0
    namelist['time_stepping']['t_max'] = 3*3600.0


    namelist['turbulence'] = {}
    namelist['turbulence']['scheme'] = 'EDMF_PrognosticTKE'
    namelist['turbulence']['EDMF_PrognosticTKE'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number'] = 1
    namelist['turbulence']['EDMF_PrognosticTKE']['constant_area'] = False
    namelist['turbulence']['EDMF_PrognosticTKE']['surface_area'] = 0.1


    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['frequency'] = 1.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Bomex_pulses'
    namelist['meta']['casename'] = 'Bomex_pulses'


    return namelist

def Bomex_cosine():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 75
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 100 / 2.5


    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['thermal_variable'] = 'thetal'

    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 30.0
    namelist['time_stepping']['t_max'] = 6*3600.0


    #namelist['turbulence'] = {}
    #namelist['turbulence']['scheme'] = 'EDMF_PrognosticTKE'
    #namelist['turbulence']['EDMF_PrognosticTKE'] = {}
    #namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number'] = 1
    #namelist['turbulence']['EDMF_PrognosticTKE']['constant_area'] = False
    #namelist['turbulence']['EDMF_PrognosticTKE']['surface_area'] = 0.1

    namelist['turbulence'] = {}
    namelist['turbulence']['scheme'] = 'EDMF_PrognosticTKE'
    namelist['turbulence']['EDMF_PrognosticTKE'] = {}
    namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number'] = 1
    namelist['turbulence']['EDMF_PrognosticTKE']['constant_area'] = False
    namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'] = 'b_w2'
    namelist['turbulence']['EDMF_PrognosticTKE']['use_local_micro'] = True
    namelist['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity'] = False
    namelist['turbulence']['EDMF_PrognosticTKE']['updraft_surface_height'] = 0.0
    namelist['turbulence']['EDMF_PrognosticTKE']['extrapolate_buoyancy'] = True


    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['frequency'] = 60.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Bomex_cosine'
    namelist['meta']['casename'] = 'Bomex_cosine'


    return namelist

def Rico():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 150
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 100 / 2.5


    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['thermal_variable'] = 'thetal'

    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0*24.0


    namelist['turbulence'] = {}
    namelist['turbulence']['scheme'] = 'EDMF_BulkSteady'
    namelist['turbulence']['EDMF_BulkSteady'] = {}
    namelist['turbulence']['EDMF_BulkSteady']['updraft_number'] = 1
    namelist['turbulence']['EDMF_BulkSteady']['constant_area'] = False
    namelist['turbulence']['EDMF_BulkSteady']['surface_area'] = 0.1


    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['frequency'] = 60.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Rico'
    namelist['meta']['casename'] = 'Rico'


    return namelist

def TRMM_LBA(): # yair

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 2000
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 10

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['thermal_variable'] = 'thetal'

    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 5.0
    namelist['time_stepping']['t_max'] = 21590.0

    namelist['turbulence'] = {}
    namelist['turbulence']['scheme'] = 'EDMF_BulkSteady'# 'EDMF_BulkSteady' #'SimilarityED'
    namelist['turbulence']['EDMF_BulkSteady'] = {}
    namelist['turbulence']['EDMF_BulkSteady']['updraft_number'] = 1
    namelist['turbulence']['EDMF_BulkSteady']['constant_area'] = False
    namelist['turbulence']['EDMF_BulkSteady']['surface_area'] = 0.1
    namelist['turbulence']['EDMF_BulkSteady']['entrainment'] = 'inverse_z'

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['frequency'] = 60.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'TRMM_LBA'
    namelist['meta']['casename'] = 'TRMM_LBA'

    return namelist

def ARM_SGP():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 220
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 20


    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['thermal_variable'] = 'thetal'

    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0 * 14.5


    namelist['turbulence'] = {}
    namelist['turbulence']['scheme'] = 'EDMF_BulkSteady'
    namelist['turbulence']['EDMF_BulkSteady'] = {}
    namelist['turbulence']['EDMF_BulkSteady']['updraft_number'] = 1
    namelist['turbulence']['EDMF_BulkSteady']['constant_area'] = False
    namelist['turbulence']['EDMF_BulkSteady']['surface_area'] = 0.1


    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['frequency'] = 60.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'ARM_SGP'
    namelist['meta']['casename'] = 'ARM_SGP'


    return namelist

def SCMS():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 250
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 20


    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['thermal_variable'] = 'thetal'

    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0 * 12.0


    namelist['turbulence'] = {}
    namelist['turbulence']['scheme'] = 'EDMF_BulkSteady'
    namelist['turbulence']['EDMF_BulkSteady'] = {}
    namelist['turbulence']['EDMF_BulkSteady']['updraft_number'] = 1
    namelist['turbulence']['EDMF_BulkSteady']['constant_area'] = False
    namelist['turbulence']['EDMF_BulkSteady']['surface_area'] = 0.1


    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['frequency'] = 60.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'SCMS'
    namelist['meta']['casename'] = 'SCMS'


    return namelist

def GATE_III(): # yair
    # adopted from: "Large eddy simulation of Maritime Deep Tropical Convection",
    # By Khairoutdinov et al (2009)  JAMES, vol. 1, article #15
    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 2700
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 10

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['thermal_variable'] = 'thetal'

    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 5.0
    namelist['time_stepping']['t_max'] = 3600.0 * 24.0

    namelist['turbulence'] = {}
    namelist['turbulence']['scheme'] = 'EDMF_BulkSteady'# 'EDMF_BulkSteady' #'SimilarityED'
    namelist['turbulence']['EDMF_BulkSteady'] = {}
    namelist['turbulence']['EDMF_BulkSteady']['updraft_number'] = 1
    namelist['turbulence']['EDMF_BulkSteady']['constant_area'] = False
    namelist['turbulence']['EDMF_BulkSteady']['surface_area'] = 0.1
    namelist['turbulence']['EDMF_BulkSteady']['entrainment'] = 'inverse_z'

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['frequency'] = 60.0

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'GATE_III'
    namelist['meta']['casename'] = 'GATE_III'

    return namelist

def write_file(namelist):

    try:
        type(namelist['meta']['simname'])
    except:
        print('Casename not specified in namelist dictionary!')
        print('FatalError')
        exit()

    namelist['meta']['uuid'] = str(uuid.uuid4())

    fh = open(namelist['meta']['simname'] + '.in', 'w')
    pprint.pprint(namelist)
    json.dump(namelist, fh, sort_keys=True, indent=4)
    fh.close()

    return


if __name__ == '__main__':
    main()