import argparse
import json


def main():
    # Parse information from the command line
    parser = argparse.ArgumentParser(prog='SCAMPy')
    parser.add_argument("namelist")
    args = parser.parse_args()

    file_namelist = open(args.namelist).read()
    namelist = json.loads(file_namelist)
    del file_namelist

    main1d(namelist)

    return

def main1d(namelist):
    import Simulation1d
    Simulation = Simulation1d.Simulation1d(namelist)
    Simulation.initialize(namelist)
    Simulation.run()

    return

if __name__ == "__main__":
    main()





