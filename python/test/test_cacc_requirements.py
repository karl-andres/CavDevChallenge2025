from pathlib import Path
import sys
import pytest
import pandas as pd

PROJECT_BASE = Path(__file__).parent.parent.parent

dependencies = [
    PROJECT_BASE / 'python' / 'besee_core',
    PROJECT_BASE / 'python' / 'metrics'
]

for dep in dependencies: sys.path.append(str(dep))

from conftest import get_besee_logs_for_scenario_list
from metrics import rms_jerk

CACC_SCENARIOS = [
    'open_straight_road',
    'straight_road_lead_vehicle_ftp'
]
CACC_SCENARIO_DATA = get_besee_logs_for_scenario_list(CACC_SCENARIOS, suppress_output=True, delete_csv=True)

@pytest.mark.parametrize('scenario', CACC_SCENARIO_DATA.values())
def test_minimum_following_distance_requirement(scenario):
    '''
    requirement under test: at all points in time, the lead vehicle shall not be closer to the 
    ego in the positive x direction than the competition-defined closest following distance from the fdcw
    '''
    
    # load data from metadata dictionary (ignore first 10 entries of csv (0.1sec) to skip setup rows)
    df: pd.DataFrame = scenario['df']
    df = df.iloc[10:]
    scenario_name = scenario['scenario_name']
    
    # IMPLEMENT ME! :)
    assert False

@pytest.mark.parametrize('scenario', CACC_SCENARIO_DATA.values())
def test_speed_error_requirement(scenario):
    '''
    requirement under test: at steady state, the relative speed error must not exceed 10%
    '''
    
    # load data from metadata dictionary (ignore first 10 entries of csv (0.1sec) to skip setup rows)
    df: pd.DataFrame = scenario['df']
    df = df.iloc[10:]
    scenario_name = scenario['scenario_name']
    
    # IMPLEMENT ME! :)
    assert False
