import subprocess
import numpy as np
import pandas as pd
import os
import yaml
from pathlib import Path
import argparse

PROJECT_BASE = Path(__file__).parent.parent.parent

def get_scenarios_from_yaml():
    with open(str(PROJECT_BASE / 'config' / 'scenarios.yaml'), 'r') as f:
        all_scenarios = yaml.safe_load(f)
        return all_scenarios

def load_loggers_csv(file_path: str):
    
    # initialize metadata dictionary
    csv_metadata = {}
    csv_metadata['path'] = file_path
    
    # get header properties of csv
    entire_csv = np.genfromtxt(file_path, dtype=None)
    for row in entire_csv:
        entries = row.split(',') if isinstance(row, str) else row.decode('utf-8').split(',')
        if entries[0] == 'Scenario': # scenario name parameter
            csv_metadata['scenario_name'] = entries[1]
            print(f'scenario name: {csv_metadata["scenario_name"]}')
        elif entries[0] == 'time': # once we hit the time column, stop parsing csv header
            break
    
    # get dataframe of timeseries data
    csv_metadata['df'] = pd.read_csv(file_path, header=1)
    print(f'available signals: {sorted(csv_metadata["df"].columns.tolist())}')
    
    return csv_metadata

def run_besee(
    scenario_name: str, 
    sim_time: float, 
    output_filename: str = '', 
    suppress_output: bool = False, 
    delete_csv: bool = False
):
    
    # determine besee executable name (besee on linux, besee.exe on windows)
    besee_exe = f"{str(PROJECT_BASE / 'bin' / 'besee')}"
    if os.name == 'nt':
        besee_exe += '.exe'
        
    # set output csv filename to default value if not specified
    if output_filename == '':
        output_filename = f'loggers_{scenario_name}'
        
    # setup command to run besee (redirect output to DEVNULL if suppress output specified)
    besee_command = f"cd {str(PROJECT_BASE / 'bin')} && {besee_exe} {scenario_name} {sim_time} {output_filename}"
    stdout = subprocess.DEVNULL if suppress_output else None
    stderr = subprocess.STDOUT if suppress_output else None
    
    # run besee command
    print(f'Executing command: {besee_command}')
    subprocess.run(besee_command, shell=True, stdout=stdout, stderr=stderr, check=True)
    
    # load data from loggers csv output
    output_csv_path = PROJECT_BASE / 'loggers' / (output_filename + '.csv')
    csv_metadata = load_loggers_csv(str(output_csv_path))
    
    # delete loggers csv if specified
    if delete_csv:
        output_csv_path.unlink()
    
    return csv_metadata
    
def main():

    # parse cmd line argument to get file to process
    parser = argparse.ArgumentParser(description ='Specify arguments for running besee.')
    parser.add_argument('scenario_name', type = str, help ='scenario name')
    parser.add_argument('sim_time', type = float, nargs='?', help ='duration to run simulation (sec)', default=100)
    parser.add_argument('file_name', type = str, nargs='?', help ='output csv filename', default='')
    args = parser.parse_args()
    
    scenarios = get_scenarios_from_yaml()
    if args.scenario_name not in scenarios.keys():
        print(f'WARNING: NO SCENARIO CALLED {args.scenario_name}. PLEASE SELECT FROM ONE OF {list(scenarios.keys())}')
        return
    
    run_besee(args.scenario_name, args.sim_time, args.file_name)
    
if __name__ == '__main__':
    main()