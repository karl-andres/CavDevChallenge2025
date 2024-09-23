import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from typing import List
from pathlib import Path

PROJECT_BASE = Path(__file__).parent.parent.parent

dependencies = [
    PROJECT_BASE / 'python' / 'besee_core',
    PROJECT_BASE / 'python' / 'metrics'
]

for dep in dependencies: sys.path.append(str(dep))

from besee_utils import load_loggers_csv

def timeseries_plot(df_name: str, df: pd.DataFrame, conf: List):
    
    fig, axes = plt.subplots(len(conf), sharex=True)

    time = df['time'].values

    plt_num = 0
    for signal_set in conf:
        for signal in signal_set:
            axes[plt_num].plot(time, df[signal].values, marker='.', label=signal, markersize=2)
        plt_num += 1

    for ax in axes:
        ax.grid()
        ax.legend()

    fig.set_size_inches(10, 10)
    fig.suptitle(f'LOGGERS VIZ: {df_name}')
    fig.supxlabel('Time (sec)')
    fig.tight_layout()

    plt.show()

def main():

    # parse cmd line argument to get file to process
    parser = argparse.ArgumentParser(description ='Specify csv to visualize.')
    parser.add_argument('file_path', type = str, help ='csv to process')
    args = parser.parse_args()
    
    # setup your plotter config
    signal_conf = [
        ['ego_speed'],
        ['ego_soc'],
        ['cav_torque_req'],
        ['cav_brake_req'],
        ['ego_acceleration'],
        #['ACTOR_lead_x'],
        #['ACTOR_lead_y'],
        #['ACTOR_lead_heading'],
        #['ACTOR_ego_x'],
        #['ACTOR_lead_speed']
    ]
    
    # process df and plot signals
    metadata = load_loggers_csv(args.file_path)
    df = metadata['df']
    timeseries_plot(args.file_path, metadata['df'], signal_conf)

if __name__ == '__main__':
    main()
