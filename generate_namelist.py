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
    elif case_name == 'Soares':
        namelist = Soares()
    elif case_name == 'Rico':
        namelist = Rico()
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





def Rico():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 1
    namelist['grid']['nz'] = 100
    namelist['grid']['gw'] = 2
    namelist['grid']['dz'] = 40.0


    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['thermal_variable'] = 'thetal'

    namelist['time_stepping'] = {}
    namelist['time_stepping']['dt'] = 20.0
    namelist['time_stepping']['t_max'] = 86400.0


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
    namelist['meta']['simname'] = 'Rico'
    namelist['meta']['casename'] = 'Rico'


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