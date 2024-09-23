from typing import List, Dict
import math
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from path_utils import Path2D
import copy

class LaneSegment:
    '''base class to store common data/logic for lane segments'''

    def __init__(self):
        pass

class StraightLaneSegment(LaneSegment):

    # give straight road segment a length
    def __init__(self, length_m: float):
        super().__init__()
        self.length = length_m

    # get path with a specified resolution to define the road segment
    def get_path(self, x: float = 0, y: float = 0, heading: float = 0, resolution: float = 0.1):
        '''create a set of waypoints defining the straight line region and transform by initial x, y, heading'''
        x_points = np.arange(0, self.length, resolution)
        y_points = np.zeros(len(x_points))
        headings = np.zeros(len(x_points))
        waypoints = np.vstack([x_points, y_points, headings]).T
        path = Path2D(waypoints)
        path.transform_path(x, y, heading)
        return path

class ArcLaneSegment(LaneSegment):

    def __init__(self, radius_m: float, rotation_pct: float, clockwise: bool):
        super().__init__()
        self.radius = radius_m
        self.rotation_pct = rotation_pct
        self.clockwise = clockwise

    # get 2d array of x,y,heading with a specified resolution to define the road segment
    def get_path(self, x: float = 0, y: float = 0, heading: float = 0, resolution: float = 0.001):
        '''get waypoints defining the arc road segment and transform by some initial x, y, heading'''
        sign = -1 if self.clockwise else 1
        angles = np.arange(0, 2*math.pi*self.rotation_pct, resolution)

        x_points = self.radius * np.sin(angles)
        y_points = sign * (self.radius - self.radius*np.cos(angles))
        headings = sign * angles
        waypoints = np.vstack([x_points, y_points, headings]).T
        path = Path2D(waypoints)
        path.transform_path(x, y, heading)
        return path



class SineLaneSegment(LaneSegment):

    def __init__(self, period_m: float, amplitude_m: float, offset_m: float, num_cycles: float):
        super().__init__()
        self.period_m = period_m
        self.amplitude_m = amplitude_m
        self.offset_m = offset_m
        self.num_cycles = num_cycles

    def get_path(self, x: float = 0, y: float = 0, heading: float = 0, resolution: float = 0.1):
        '''get waypoints of sine lane segment and transform by some initial x, y, heading'''

        # x values from 0 to number of cycles times period (add resolution to end to include last value)
        x_points = np.arange(0, self.num_cycles*self.period_m + resolution, resolution)

        # calculate sine wave for y points and offset by initial y value (to make sure road is not discontinuous)
        y_init = self.amplitude_m * math.sin(2 * math.pi * -self.offset_m / self.period_m)
        y_points = self.amplitude_m * np.sin( (2 * math.pi / self.period_m) * (x_points - self.offset_m) ) - y_init

        # calculate by taking the arctan of the derivative of y points
        headings = np.arctan( 2 * math.pi * self.amplitude_m / self.period_m * np.cos( 2 * math.pi / self.period_m * ( x_points - self.offset_m ) ) )
        
        waypoints = np.vstack([x_points, y_points, headings]).T
        path = Path2D(waypoints)
        path.transform_path(x, y, heading)
        return path

class LaneDefinition:

    def __init__(self, segments: List[LaneSegment], num_lanes=1, lane_width_m=3.5):
        self.segments = segments
        self.num_lanes = num_lanes
        self.lane_width_m = lane_width_m
        self.lane_dict: Dict[int, Path2D] = {}
        self.lane_boundary_dict: Dict[int, Path2D] = {}
        base_path = self.get_base_path()

        # create a waypoint array entry in the lane id dict with lanes starting with a y offset of lane width away from each other
        # note: lane id 0 is the rightmost lane, every subsequent lane id is one lane to the left
        self.lane_dict[0] = base_path
        for lane_id in range(1, num_lanes):
            self.lane_dict[lane_id] = copy.deepcopy(base_path)
            self.lane_dict[lane_id].transform_path_along_heading(lane_width_m*lane_id)

        # create an entry in lane boundary dict for each centerlane with lane boundary id 0 being rightmost lane boundary
        self.lane_boundary_dict[0] = self.get_boundary_waypoints(0, left=False)
        for lane_id in self.lane_dict.keys():
            self.lane_boundary_dict[lane_id + 1] = self.get_boundary_waypoints(lane_id)

    def get_base_path(self):
        '''get path of the centerlane defined by the lane segment waypoints'''
        
        # get waypoints for a segment, update state, and iterate through all segments to get all waypoints
        path = Path2D(np.array([[0,0,0]]))
        for segment in self.segments:
            path.concat_path(segment.get_path())
        return path
    
    def get_boundary_waypoints(self, lane_id, left=True):
        '''get waypoint array of lane boundaries (left or right)'''
        boundary = copy.deepcopy(self.lane_dict[lane_id])
        boundary.transform_path_along_heading(self.lane_width_m/2, left)
        return boundary

    def draw(self, fig=None, ax=None, show=True):
        '''plot lanes from birds-eye view'''

        if fig is None:
            fig, ax = plt.subplots()

        for lane in self.lane_dict.values():
            lane.draw(fig, ax, show_headings=False, color='red', linewidth=1, markersize=1, linestyle=':')

        for lane_boundary in self.lane_boundary_dict.values():
            lane_boundary.draw(fig, ax, color='blue', markersize=1, linewidth=2, linestyle='--')

        if show:
            plt.show()
