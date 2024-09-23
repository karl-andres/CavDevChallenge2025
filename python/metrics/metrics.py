import numpy as np
import pandas as pd
import math

def calculate_timeseries_jerk(df: pd.DataFrame):
    
    times = df['time']
    accs = df['ego_acceleration']
    
    jerks = accs.diff().values / times.diff().values
    jerks[0] = 0
    
    return jerks

def rms_jerk(df: pd.DataFrame, min_speed_mps: float = 0):
    
    speeds = df['ego_speed']
    jerks = calculate_timeseries_jerk(df)
    
    jerks_filt = jerks[speeds.values >= min_speed_mps]
    
    if (len(jerks_filt) == 0):
        print('JERK METRIC WARNING: THERE ARE NO ENTRIES IN THE FILTERED JERK ARRAY. RETURNING INFINITY.')
        return math.inf
    
    jerks_filt = jerks_filt[1:]
    
    return np.sqrt(np.mean(np.square(jerks_filt)))
