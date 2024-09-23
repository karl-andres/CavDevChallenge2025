import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import patches

from path_utils import Path2D

class ActorBase:

    def __init__(self, name: str, length_m: float, width_m: float):
        self.name = name
        self.length_m = length_m
        self.width_m = width_m
        self.state = np.array([0,0,0]) # x,y,heading

    def draw(self, fig=None, ax=None, show=False):

        if fig is None:
            fig, ax = plt.subplots()

        rect = patches.Rectangle(
            xy=(-self.state[1] - self.width_m / 2, self.state[0] - self.length_m / 2),
            width=self.width_m,
            height=self.length_m,
            angle=math.degrees(self.state[2]),
            rotation_point='center',
            fill=True,
            color='green',
            zorder=100
        )

        ax.add_patch(rect)

        if show:
            plt.show()
            
    def get_rect(self):
        rect = patches.Rectangle(
            xy=(-self.state[1] - self.width_m / 2, self.state[0] - self.length_m / 2),
            width=self.width_m,
            height=self.length_m,
            angle=math.degrees(self.state[2]),
            rotation_point='center',
            fill=True,
            color='green',
            zorder=100)
        return rect

class PathFollowingActor(ActorBase):
    
    def __init__(self, name: str, path: Path2D, speed_mps = 0, init_dist = 0, length_m: float = 4, width_m: float = 2):
        super().__init__(name, length_m, width_m)
        self.path = path
        self.speed_mps = speed_mps
        self.dist_along_path = init_dist
        self.state = self.path.distance_to_waypoint(self.dist_along_path)

    def update(self, dt):
        '''update actor's state after time dt seconds'''
        self.dist_along_path += self.speed_mps*dt
        self.state = self.path.distance_to_waypoint(self.dist_along_path)