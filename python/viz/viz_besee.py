from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd, numpy as np
import yaml
import argparse
import sys

PROJECT_BASE = Path(__file__).parent.parent.parent

dependencies = [
    PROJECT_BASE / 'python' / 'besee_core',
    PROJECT_BASE / 'python' / 'metrics'
]

for dep in dependencies: sys.path.append(str(dep))

from lane_utils import LaneDefinition, StraightLaneSegment, ArcLaneSegment
from actor_utils import ActorBase
from besee_utils import load_loggers_csv

class BeseeViz:
    
    def __init__(self, csv_path: Path, frame_rate: int=20, time_ratio: float=1):
        self.csv_path = Path(csv_path)
        self.lane_definition = None
        self.actor_dict = {}
        self.ego = None
        self.signal_df = None
        self.fig = None
        self.ax = None
        self.frame_rate = frame_rate
        self.dt = 0.01 # this is overwritten with the value determined from time deltas of csv
        self.time = 0
        self.time_ratio = time_ratio
        
        self.fig = plt.figure()
        self.ax = plt.axes()
        
        self.plottable_actors = {}
        
        self.parse_config_files()
        
        self.lane_definition.draw(self.fig, self.ax, show=False)
        
    def parse_config_files(self):
        
        metadata = load_loggers_csv(str(self.csv_path))
        
        # specify location of yaml config files
        config_directory = PROJECT_BASE / 'config'
        all_actors = []
        all_roads = []
        all_scenarios = []
        
        # load yaml file contents to dictionaries
        with open(str(config_directory / 'actors.yaml'), 'r') as f:
            all_actors = yaml.safe_load(f)
            
        with open(str(config_directory / 'roads.yaml'), 'r') as f:
            all_roads = yaml.safe_load(f)
            
        with open(str(config_directory / 'scenarios.yaml'), 'r') as f:
            all_scenarios = yaml.safe_load(f)
            
        # load relevant scenario from scenario list
        scenario = all_scenarios[metadata['scenario_name']]
        
        lane_def = all_roads[scenario['lanes']]
        actors_list = scenario['actors']
        
        # construct LaneDefinition object
        lane_segments = []
        
        for seg_dict in lane_def['lane_segments']:
            
            seg_type = list(seg_dict.keys())[0]
            seg_properties = list(seg_dict.values())[0]
            
            if seg_type == 'straight':
                lane_segments.append(
                    StraightLaneSegment(float(seg_properties['length']))
                )
                
            elif seg_type == 'arc':
                lane_segments.append(
                    ArcLaneSegment(
                        float(seg_properties['radius']),
                        float(seg_properties['rotations']),
                        float(seg_properties['clockwise'])
                    )
                )
                
            else:
                print(f'{seg_type} not yet supported in besee viz :(')
                
        self.lane_definition = LaneDefinition(
            lane_segments,
            int(lane_def['num_lanes']),
            float(lane_def['lane_width'])
        )
        
        # loop through all actors and create actor objects
        for actor_dict in actors_list:
            
            actor_name = list(actor_dict.keys())[0]
            actor_def = list(actor_dict.values())[0]
            
            actor_dimensions = all_actors[actor_def['make']]
            self.actor_dict[actor_name] = ActorBase(
                actor_name,
                actor_dimensions['length'],
                actor_dimensions['width']
            )
            
        # create ego actor and add to actors dict (it is a special actor!)
        ego_dimensions = all_actors[scenario['ego']['make']]
        self.ego = ActorBase(
            'ego',
            ego_dimensions['length'],
            ego_dimensions['width']
        )
        self.actor_dict['ego'] = self.ego
            
        # load besee config file and save to a dataframe
        self.signal_df = metadata['df']
        self.dt = np.diff(self.signal_df['time'])[0]
        
    def update_actor_states(self, time_s):
        
        # determine current index to query for dataframe
        print(time_s)
        idx = int(time_s / self.dt)
       
        # update actor states by their position recorded in csv
        for actor_name, actor in self.actor_dict.items():
            x = self.signal_df[f'ACTOR_{actor_name}_x'][idx]
            y = self.signal_df[f'ACTOR_{actor_name}_y'][idx]
            heading = self.signal_df[f'ACTOR_{actor_name}_heading'][idx]
            actor.state = np.array([x, y, heading])
            print(time_s, actor_name, actor.state)
    
    def init_animation(self):
        
        # initialize actor positions
        self.update_actor_states(0)
        
        # construct dictionary of rectangles for mpl to draw
        self.plottable_actors = {}
        
        for actor_name, actor in self.actor_dict.items():
            self.plottable_actors[actor_name] = actor.get_rect()
            if actor_name == 'ego':
                    self.plottable_actors[actor_name].set(color='maroon')
                
        print('init', self.plottable_actors)
        return [self.ax.add_patch(rect) for rect in list(self.plottable_actors.values())]
        
    def step_animation(self,i):
        
        self.time += (1 / self.frame_rate)
        
        self.update_actor_states(self.time)
        for actor_name, actor in self.actor_dict.items():
            temp_rect = actor.get_rect()
            self.plottable_actors[actor_name].set_xy(temp_rect.get_xy())
            self.plottable_actors[actor_name].set_angle(temp_rect.get_angle())
            
        ego_x = self.plottable_actors['ego'].get_x()
        ego_y = self.plottable_actors['ego'].get_y()
            
        self.ax.set_xlim(ego_x - 55, ego_x + 55)
        self.ax.set_ylim(ego_y - 10, ego_y + 100)
        
        self.ax.set_title(f'Visualizing {self.csv_path.name} @ t = {self.time:.2f} sec!')
        
        return list(self.plottable_actors.values())
        
    def animate(self):
        
        print(f'numrows: {float(self.signal_df["time"].values[-1])}')
        print(f'framerate: {self.frame_rate}')
        print(f'dt: {self.dt}')
        print(f'numframes: {int(self.signal_df["time"].values[-1] / (1 / self.frame_rate))}')
        
        anim = animation.FuncAnimation(
            self.fig, 
            self.step_animation, 
            init_func=self.init_animation,
            frames=int(self.signal_df['time'].values[-1] / (1 / self.frame_rate)),
            interval=(1 / (self.frame_rate * self.time_ratio)) * 1000,
            blit=False,
            repeat=False
        )
        
        plt.show()
        
def main():

    # parse cmd line argument to get file to process
    parser = argparse.ArgumentParser(description ='Specify csv to visualize.')
    parser.add_argument('file_path', type = str, help ='csv to process')
    parser.add_argument('time_ratio', type = float, nargs='?', help ='speed at which visualizer will run (1 = real time, smaller = slower, larger = faster)', default=1)
    args = parser.parse_args()
    
    viz = BeseeViz(Path(args.file_path), time_ratio=args.time_ratio)
    viz.animate()

if __name__ == '__main__':
    main()