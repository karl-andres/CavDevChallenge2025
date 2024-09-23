import numpy as np
from scipy import interpolate
import math, copy
import matplotlib.pyplot as plt

def get_rotation_matrix(heading: float):
    '''given a heading angle in radians, get 2d rotation matrix'''
    return np.array([[math.cos(heading), -math.sin(heading)],
                     [math.sin(heading), math.cos(heading)]])

class Path2D:

    def __init__(self, waypoints: np.ndarray):
        '''takes in a 2d array of N waypoints (rows) with x, y, heading'''

        # assumptions: array's shape is Nx3
        assert waypoints.shape[1] == 3, "Waypoints array must be Nx3 in the form [x,y,heading]"

        self.waypoints = waypoints

        self.update_distance()

    def update_distance(self):
        '''update linear distance along path and create interpolators for x, y, heading'''

        # calculate x and y deltas, then calculated cumulative distance along the path
        deltas = np.diff(self.waypoints[:,0:2], axis=0, prepend=0)
        deltas[0,0] = 0
        deltas[0,1] = 0
        x_deltas = deltas[:,0]
        y_deltas = deltas[:,1]
        self.dist = np.cumsum(np.sqrt(np.square(x_deltas) + np.square(y_deltas)))

        # add dist column to waypoints array
        # self.waypoints = np.hstack([self.waypoints, dist])

        # create interpolator functions
        self.x_interpolator = interpolate.interp1d(self.dist, self.waypoints[:,0])
        self.y_interpolator = interpolate.interp1d(self.dist, self.waypoints[:,1])
        self.heading_interpolator = interpolate.interp1d(self.dist, self.waypoints[:,2])

    def distance_to_waypoint(self, distance_m):
        '''linearly interpolate x, y, and heading based on a distance along the path'''
        return np.array([self.x_interpolator(distance_m),
                         self.y_interpolator(distance_m),
                         self.heading_interpolator(distance_m)])
    
    def transform_path(self, x=0, y=0, heading=0):
        '''rotate path by specified heading, then translate it by specified x and y distance'''

        # save some computation if the heading/transforms are 0
        if heading != 0:
            # rotate by heading about the origin
            rot_mat = get_rotation_matrix(-heading)
            self.waypoints[:,0:2] = np.matmul(self.waypoints[:,0:2], rot_mat)
            self.waypoints[:,2] += heading

        # translate path by x
        if x != 0:
            self.waypoints[:,0] = self.waypoints[:,0] + x

        # translate path by y
        if y != 0:
            self.waypoints[:,1] = self.waypoints[:,1] + y

        self.update_distance()

    def transform_path_along_heading(self, offset_m, left: bool=True):
        '''offset a lane by a certain distance (left or right) and transform lane to handle any error with non-zero headings'''
        
        # get waypoint headings of centerlane and construct vector of specified length in that direction
        headings = self.waypoints[:,2]
        lane_offsets_x = (offset_m) * np.cos(headings)
        lane_offsets_y = (offset_m) * np.sin(headings)
        lane_offsets = np.vstack([lane_offsets_x, lane_offsets_y]).T

        # if left, rotate vectors by pi/2. if right, rotate by -pi/2.
        sign = 1 if left else -1
        rot_mat = get_rotation_matrix(-sign * (math.pi / 2))
        lane_offsets = np.matmul(lane_offsets, rot_mat)

        # add a dummy column of zeros (headings are not changed (?))
        lane_offsets = np.append(lane_offsets, np.zeros(lane_offsets.shape[0])[...,None], 1)

        self.waypoints += lane_offsets
        self.update_distance()

    def concat_path(self, path, path_mutable=True):
        '''concatenate input path to the end of current path. if input path needs to be preserved, set path_mutable to False'''

        # copy input path if it needs to be preserved
        if not path_mutable:
            path = np.copy(path)

        # transform input path to be continuous with the end of the existing path
        curr_x, curr_y, curr_heading = self.waypoints[-1,:]
        path.transform_path(curr_x, curr_y, curr_heading)

        self.waypoints = np.vstack([self.waypoints, path.waypoints])

    def draw(self, fig=None, ax=None, show=False, show_waypoints=True, show_headings=False, **kwargs):
        '''visualize path with matplotlib. either pass in fig to draw on, otherwise it will be created'''

        # if no fig/ax passed in, create them
        if fig is None:
            fig, ax = plt.subplots()

        if show_waypoints:
            kwargs['marker'] = 'o'

        ax.plot(-self.waypoints[:,1], self.waypoints[:,0], **kwargs)
        
        if show_headings:
            for waypoint in self.waypoints:
                vector_len = 0.1
                ax.arrow(-waypoint[1], waypoint[0], -vector_len*math.sin(waypoint[2]), vector_len*math.cos(waypoint[2]))

        if show:
            plt.show()