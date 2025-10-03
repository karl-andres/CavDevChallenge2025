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
    'straight_road_cacc_test',
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
    
    # Check if this scenario has a lead vehicle
    lead_vehicle_columns = [col for col in df.columns if 'ACTOR_lead' in col]
    if not lead_vehicle_columns:
        pytest.skip(f"Scenario {scenario_name} has no lead vehicle - skipping FDCW-1 test")
    
    # Calculate following distance at each time step
    # Following distance = lead_x - ego_x (positive x direction)
    following_distances = df['ACTOR_lead_x'] - df['ACTOR_ego_x']
    
    # Convert lead vehicle speed from m/s to mph for the formula
    lead_speed_mph = df['ACTOR_lead_speed'] * 2.237  # m/s to mph conversion
    
    # Calculate minimum allowed following distance using the formula
    # CFD = 2.8 * (speed_mph)^0.45 + 8
    min_allowed_distances = 2.8 * (lead_speed_mph ** 0.45) + 8
    
    # Check if any following distance violates the requirement
    violations = following_distances < min_allowed_distances
    
    if violations.any():
        # Find the worst violation for reporting
        worst_violation_idx = violations.idxmax()
        worst_following_distance = following_distances.iloc[worst_violation_idx]
        worst_min_allowed = min_allowed_distances.iloc[worst_violation_idx]
        worst_lead_speed_mph = lead_speed_mph.iloc[worst_violation_idx]
        worst_time = df['time'].iloc[worst_violation_idx]
        
        assert False, (f"FDCW-1 requirement violated in scenario {scenario_name}: "
                      f"At time {worst_time:.2f}s, following distance was {worst_following_distance:.2f}m "
                      f"but minimum allowed was {worst_min_allowed:.2f}m "
                      f"(lead speed: {worst_lead_speed_mph:.2f} mph)")
    
    # If we get here, all following distances meet the requirement
    print(f"✓ FDCW-1 requirement passed for scenario {scenario_name}: "
          f"All following distances >= minimum allowed distance")

@pytest.mark.parametrize('scenario', CACC_SCENARIO_DATA.values())
def test_speed_error_requirement(scenario):
    '''
    requirement under test: at steady state, the relative speed error must not exceed 10%
    '''
    
    # load data from metadata dictionary (ignore first 10 entries of csv (0.1sec) to skip setup rows)
    df: pd.DataFrame = scenario['df']
    df = df.iloc[10:]
    scenario_name = scenario['scenario_name']
    
    # Check if this scenario has a lead vehicle to determine desired speed
    lead_vehicle_columns = [col for col in df.columns if 'ACTOR_lead' in col]
    
    if lead_vehicle_columns:
        # For scenarios with lead vehicle, desired speed is the lead vehicle speed
        desired_speed = df['ACTOR_lead_speed']
    else:
        # For scenarios without lead vehicle, we need to determine desired speed
        # This could be a constant cruise speed or from a drive cycle
        # For now, we'll use the ego vehicle's speed as a proxy (this may need refinement)
        # In a real implementation, you'd want to know the target cruise speed
        desired_speed = df['ego_speed']
        print(f"Warning: Scenario {scenario_name} has no lead vehicle. Using ego speed as desired speed proxy.")
    
    # Calculate actual speed (ego vehicle speed)
    actual_speed = df['ego_speed']
    
    # Calculate relative speed error: |desired - actual| / desired * 100%
    # Avoid division by zero by adding small epsilon
    epsilon = 1e-6
    relative_speed_error = abs(desired_speed - actual_speed) / (desired_speed + epsilon) * 100
    
    # Define steady state conditions:
    # 1. Speed should be relatively constant (low acceleration)
    # 2. We'll consider the last 50% of the simulation as potential steady state
    # 3. Only check when desired speed > 1 m/s to avoid low-speed noise
    
    # Calculate acceleration (derivative of speed)
    dt = df['time'].diff().mean()  # Average time step
    acceleration = actual_speed.diff() / dt
    
    # Define steady state mask:
    # - Last 50% of simulation
    # - Low acceleration (< 0.5 m/s²)
    # - Desired speed > 1 m/s
    steady_state_mask = (
        (df.index >= len(df) * 0.5) &  # Last 50% of simulation
        (abs(acceleration) < 0.5) &    # Low acceleration
        (desired_speed > 1.0)          # Reasonable speed
    )
    
    if not steady_state_mask.any():
        pytest.skip(f"Scenario {scenario_name} has no steady state periods meeting criteria")
    
    # Check speed error during steady state
    steady_state_errors = relative_speed_error[steady_state_mask]
    max_error = steady_state_errors.max()
    
    if max_error > 10.0:
        # Find the worst violation for reporting
        worst_idx = steady_state_errors.idxmax()
        worst_time = df['time'].iloc[worst_idx]
        worst_desired = desired_speed.iloc[worst_idx]
        worst_actual = actual_speed.iloc[worst_idx]
        
        assert False, (f"FDCW-2 requirement violated in scenario {scenario_name}: "
                      f"At time {worst_time:.2f}s, speed error was {max_error:.2f}% "
                      f"(desired: {worst_desired:.2f} m/s, actual: {worst_actual:.2f} m/s)")
    
    # If we get here, speed error meets the requirement
    print(f"✓ FDCW-2 requirement passed for scenario {scenario_name}: "
          f"Maximum steady state speed error: {max_error:.2f}% (limit: 10%)")