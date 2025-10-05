#!/usr/bin/env python3
"""
Drive Cycle Generator for CACC Development Challenge

This script creates custom drive cycles that meet the CACC requirements:
- CACC-Planning-FDCW-1: Minimum following distance requirement
- CACC-Planning-FDCW-2: Speed error requirement (≤10% at steady state)

The generated drive cycle will be saved as a CSV file in the config/drive_cycle/ directory
and can be used in scenarios.yaml by referencing the filename (without .csv extension).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Project paths
PROJECT_BASE = Path(__file__).parent.parent
DRIVE_CYCLE_DIR = PROJECT_BASE / "config" / "drive_cycle"

# setting a function for the drive cycle
def create_cacc_optimized_drive_cycle(
    duration_sec: float = 100.0,
    dt: float = 0.02,
    max_speed_mps: float = 30.0,  # ~67 mph
    min_speed_mps: float = 5.0,   # ~11 mph
    scenario_type: str = "mixed"
) -> pd.DataFrame:
    """
    Create a drive cycle optimized for CACC testing.
    
    Args:
        duration_sec: Total duration of the drive cycle
        dt: Time step in seconds (0.02s = 50Hz, matches BESEE)
        max_speed_mps: Maximum speed in m/s
        min_speed_mps: Minimum speed in m/s
        scenario_type: Type of scenario ("highway", "city", "mixed") 
    
    Returns:
        DataFrame with 'Time (s)' and 'Speed (m/s)' columns
    """
    
    # Create time vector
    time = np.arange(0, duration_sec, dt)
    n_points = len(time)
    
    # creates scenario based on road
    if scenario_type == "highway":
        # Highway scenario: mostly steady speeds with gradual changes
        speeds = create_highway_profile(time, max_speed_mps, min_speed_mps)
    elif scenario_type == "city":
        # City scenario: frequent stops and starts
        speeds = create_city_profile(time, max_speed_mps, min_speed_mps)
    else:  # mixed
        # Mixed scenario: combination of highway and city driving
        speeds = create_mixed_profile(time, max_speed_mps, min_speed_mps)
    
    # Ensure speeds are within bounds
    speeds = np.clip(speeds, 0, max_speed_mps)
    
    # Based on the time and speeds above, it will be initalized in a dataframe
    df = pd.DataFrame({
        'Time (s)': time,
        'Speed (m/s)': speeds
    })
    
    return df

def create_highway_profile(time: np.ndarray, max_speed: float, min_speed: float) -> np.ndarray:
    """Create a highway driving profile with steady speeds and gradual changes."""
    
    # Makes all the speeds 0 in the array, basically initializing values
    speeds = np.zeros_like(time)
    
    # Phase 1: Acceleration to cruising speed (0-20s)
    # Makes sure its to 80% of the max speed since its on the highway
    accel_phase = time <= 20
    speeds[accel_phase] = np.linspace(0, max_speed * 0.8, np.sum(accel_phase))
    
    # Phase 2: Steady cruising (20-40s)
    # Now the speeds have shifted from rapidly moving to slightly
    cruise_phase = (time > 20) & (time <= 40)
    # speed is constant, no change like linspace
    speeds[cruise_phase] = max_speed * 0.8
    
    # Phase 3: Speed adjustment for following (40-60s)
    adjust_phase = (time > 40) & (time <= 60)
    speeds[adjust_phase] = max_speed * 0.8 + 5 * np.sin(2 * np.pi * (time[adjust_phase] - 40) / 20)
    
    # Phase 4: Steady following (60-80s)
    follow_phase = (time > 60) & (time <= 80)
    speeds[follow_phase] = max_speed * 0.7
    
    # Phase 5: Deceleration (80-100s)
    decel_phase = time > 80
    speeds[decel_phase] = np.linspace(max_speed * 0.7, min_speed, np.sum(decel_phase))
    
    return speeds

def create_city_profile(time: np.ndarray, max_speed: float, min_speed: float) -> np.ndarray:
    """Create a city driving profile with frequent stops and starts."""
    speeds = np.zeros_like(time)
    
    # Create multiple stop-and-go cycles
    cycle_duration = 20  # seconds per cycle
    n_cycles = int(len(time) / (cycle_duration / 0.02))  # 0.02s timestep
    
    for i in range(n_cycles):
        start_idx = int(i * cycle_duration / 0.02)
        end_idx = int((i + 1) * cycle_duration / 0.02)
        end_idx = min(end_idx, len(time))
        
        if start_idx >= len(time):
            break
            
        cycle_time = time[start_idx:end_idx] - time[start_idx]
        
        # Each cycle: accelerate, cruise, decelerate, stop
        if len(cycle_time) > 0:
            # Acceleration (0-5s)
            accel_mask = cycle_time <= 5
            if np.any(accel_mask):
                speeds[start_idx:end_idx][accel_mask] = np.linspace(0, max_speed * 0.6, np.sum(accel_mask))
            
            # Cruise (5-10s)
            cruise_mask = (cycle_time > 5) & (cycle_time <= 10)
            if np.any(cruise_mask):
                speeds[start_idx:end_idx][cruise_mask] = max_speed * 0.6
            
            # Deceleration (10-15s)
            decel_mask = (cycle_time > 10) & (cycle_time <= 15)
            if np.any(decel_mask):
                speeds[start_idx:end_idx][decel_mask] = np.linspace(max_speed * 0.6, 0, np.sum(decel_mask))
            
            # Stop (15-20s)
            stop_mask = cycle_time > 15
            if np.any(stop_mask):
                speeds[start_idx:end_idx][stop_mask] = 0
    
    return speeds

def create_mixed_profile(time: np.ndarray, max_speed: float, min_speed: float) -> np.ndarray:
    """Create a mixed driving profile combining highway and city elements."""
    speeds = np.zeros_like(time)
    
    # Phase 1: City driving (0-30s)
    city_phase = time <= 30
    if np.any(city_phase):
        speeds[city_phase] = create_city_profile(time[city_phase], max_speed * 0.7, 0)
    
    # Phase 2: Highway acceleration (30-50s)
    hwy_accel_phase = (time > 30) & (time <= 50)
    if np.any(hwy_accel_phase):
        speeds[hwy_accel_phase] = np.linspace(0, max_speed, np.sum(hwy_accel_phase))
    
    # Phase 3: Highway cruising (50-70s)
    hwy_cruise_phase = (time > 50) & (time <= 70)
    if np.any(hwy_cruise_phase):
        speeds[hwy_cruise_phase] = max_speed
    
    # Phase 4: Highway deceleration (70-100s)
    hwy_decel_phase = time > 70
    if np.any(hwy_decel_phase):
        speeds[hwy_decel_phase] = np.linspace(max_speed, min_speed, np.sum(hwy_decel_phase))
    
    return speeds

def validate_drive_cycle(df: pd.DataFrame) -> dict:
    """
    Validate that the drive cycle meets CACC requirements.
    
    Returns:
        Dictionary with validation results
    """
    time = df['Time (s)'].values
    speed = df['Speed (m/s)'].values
    
    # Calculate acceleration
    dt = np.mean(np.diff(time))
    acceleration = np.gradient(speed, dt)
    
    # Check for reasonable acceleration limits
    max_accel = np.max(acceleration)
    max_decel = np.min(acceleration)
    
    # Check for steady state periods (for FDCW-2 requirement)
    steady_state_mask = np.abs(acceleration) < 0.5  # Low acceleration
    steady_state_periods = np.sum(steady_state_mask) / len(steady_state_mask)
    
    # Calculate speed statistics
    speed_stats = {
        'max_speed_mps': np.max(speed),
        'max_speed_mph': np.max(speed) * 2.237,
        'min_speed_mps': np.min(speed),
        'avg_speed_mps': np.mean(speed),
        'max_acceleration': max_accel,
        'max_deceleration': max_decel,
        'steady_state_percentage': steady_state_periods * 100,
        'duration_sec': time[-1] - time[0]
    }
    
    return speed_stats

def plot_drive_cycle(df: pd.DataFrame, save_path: Path = None):
    """Plot the drive cycle for visualization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Speed profile
    ax1.plot(df['Time (s)'], df['Speed (m/s)'], 'b-', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Speed (m/s)')
    ax1.set_title('Drive Cycle Speed Profile')
    ax1.grid(True, alpha=0.3)
    
    # Speed in mph
    speed_mph = df['Speed (m/s)'] * 2.237
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df['Time (s)'], speed_mph, 'r--', alpha=0.7)
    ax1_twin.set_ylabel('Speed (mph)', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # Acceleration profile
    dt = np.mean(np.diff(df['Time (s)']))
    acceleration = np.gradient(df['Speed (m/s)'], dt)
    ax2.plot(df['Time (s)'], acceleration, 'g-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Acceleration (m/s²)')
    ax2.set_title('Drive Cycle Acceleration Profile')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Create a custom drive cycle for CACC testing')
    parser.add_argument('--name', type=str, default='custom_cacc', 
                       help='Name for the drive cycle file (without .csv)')
    parser.add_argument('--duration', type=float, default=100.0,
                       help='Duration in seconds (default: 100)')
    parser.add_argument('--scenario', type=str, choices=['highway', 'city', 'mixed'], 
                       default='mixed', help='Scenario type (default: mixed)')
    parser.add_argument('--max-speed', type=float, default=30.0,
                       help='Maximum speed in m/s (default: 30.0)')
    parser.add_argument('--min-speed', type=float, default=5.0,
                       help='Minimum speed in m/s (default: 5.0)')
    parser.add_argument('--plot', action='store_true',
                       help='Show plot of the drive cycle')
    parser.add_argument('--validate', action='store_true',
                       help='Validate the drive cycle against CACC requirements')
    
    args = parser.parse_args()
    
    # Create drive cycle
    print(f"Creating {args.scenario} drive cycle...")
    df = create_cacc_optimized_drive_cycle(
        duration_sec=args.duration,
        max_speed_mps=args.max_speed,
        min_speed_mps=args.min_speed,
        scenario_type=args.scenario
    )
    
    # Save to CSV
    output_path = DRIVE_CYCLE_DIR / f"{args.name}.csv"
    df.to_csv(output_path, index=False)
    print(f"Drive cycle saved to: {output_path}")
    
    # Validate if requested
    if args.validate:
        print("\nValidating drive cycle...")
        stats = validate_drive_cycle(df)
        print(f"Duration: {stats['duration_sec']:.1f} seconds")
        print(f"Max speed: {stats['max_speed_mps']:.1f} m/s ({stats['max_speed_mph']:.1f} mph)")
        print(f"Min speed: {stats['min_speed_mps']:.1f} m/s")
        print(f"Average speed: {stats['avg_speed_mps']:.1f} m/s")
        print(f"Max acceleration: {stats['max_acceleration']:.2f} m/s²")
        print(f"Max deceleration: {stats['max_deceleration']:.2f} m/s²")
        print(f"Steady state periods: {stats['steady_state_percentage']:.1f}%")
        
        # Check CACC requirements
        print("\nCACC Requirement Checks:")
        print(f"✓ Drive cycle created with {len(df)} data points")
        print(f"✓ Time step: {np.mean(np.diff(df['Time (s)'])):.3f} seconds")
        if stats['steady_state_percentage'] > 30:
            print("✓ Sufficient steady state periods for FDCW-2 testing")
        else:
            print("⚠ Limited steady state periods - may affect FDCW-2 testing")
    
    # Plot if requested
    if args.plot:
        plot_path = DRIVE_CYCLE_DIR / f"{args.name}_plot.png"
        plot_drive_cycle(df, plot_path)
    
    print(f"\nTo use this drive cycle in scenarios.yaml, add:")
    print(f"  speed_profile: {args.name}")
    print(f"to your actor definition.")

if __name__ == "__main__":
    main()