from pathlib import Path
from typing import List, Dict, Any
import time
import sys
import multiprocessing
import multiprocessing.pool
import pytest

PROJECT_BASE = Path(__file__).parent.parent.parent

dependencies = [
    PROJECT_BASE / 'python' / 'besee_core',
    PROJECT_BASE / 'python' / 'metrics'
]

for dep in dependencies: sys.path.append(str(dep))

from besee_utils import get_scenarios_from_yaml, run_besee

def get_besee_logs_for_scenario_list(
    scenario_list: List[str], 
    delete_csv: bool = False,
    suppress_output: bool = False
) -> Dict[str, Dict[str, Any]]:
    
    # create global thread pool and a timestamp for the beginning of the test (guarantees unique filenames)
    logged_data = {}
    time_string = str(time.time()).replace('.', '')
    
    # local scoped helper function to run besee and append metadata to dict
    # necessary to separate out like this to enable parallelism with ThreadPool
    def run_besee_and_append_metadata(
        scenario_name: str
    ):
        metadata = run_besee(
            scenario_name,
            100, # TODO: update this time parameter to be dynamic (maybe a property of the scenario config)
            'TEST_' + time_string + '_' + scenario_name,
            delete_csv=delete_csv,
            suppress_output=suppress_output
        )
        logged_data[scenario_name] = metadata
        
    # create a thread pool with n_workers = CPU count / 2
    pool_size = multiprocessing.cpu_count() // 2 # uses half your thread count to not blow up your computer
    print(f'running besee on {pool_size} cores!')
    tp = multiprocessing.pool.ThreadPool(pool_size)
    
    # assign all workers to the given scenario list
    tp.map(run_besee_and_append_metadata, scenario_list)
        
    # return csv metadata dict
    return logged_data

@pytest.fixture(scope='session')
def all_besee_scenarios():
    
    all_scenarios = get_scenarios_from_yaml()
    return get_besee_logs_for_scenario_list(all_scenarios)
