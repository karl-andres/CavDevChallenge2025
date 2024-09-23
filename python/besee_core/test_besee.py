from lane_utils import StraightLaneSegment, ArcLaneSegment, LaneDefinition, SineLaneSegment
from actor_utils import PathFollowingActor
import matplotlib.pyplot as plt

def test_straight_road():

    straight_segment = StraightLaneSegment(50)
    lane = LaneDefinition([straight_segment], num_lanes=4)
    lane.draw()

def test_matt_track(clockwise: bool):

    straight_segment = StraightLaneSegment(50)
    arc_segment = ArcLaneSegment(radius_m=50, rotation_pct=.5, clockwise=clockwise)

    lane = LaneDefinition([
        straight_segment,
        arc_segment,
        straight_segment,
        arc_segment
    ],
    num_lanes=4)

    lane.draw()

def test_sine_wobbly_road():

    straight_segment = StraightLaneSegment(10)
    wobbly_segment = SineLaneSegment(period_m=50, amplitude_m=5, offset_m=12.5, num_cycles=1)

    lane = LaneDefinition([
        straight_segment,
        wobbly_segment,
        straight_segment
    ],
    num_lanes=4)

    lane.draw()

def test_arc_wobbly_road(bend_radius_m: float=50, rot_pct=0.125):

    straight_segment = StraightLaneSegment(10)
    arc_left = ArcLaneSegment(radius_m=bend_radius_m, rotation_pct=rot_pct, clockwise=False)
    arc_right = ArcLaneSegment(radius_m=bend_radius_m, rotation_pct=rot_pct, clockwise=True)

    lane = LaneDefinition([
        straight_segment,
        arc_right,
        straight_segment,
        arc_left,
        straight_segment,
        arc_left,
        straight_segment,
        arc_right,
        straight_segment
    ],
    num_lanes=4)

    lane.draw()

def test_everything():
    straight_segment = StraightLaneSegment(10)
    arc_left = ArcLaneSegment(radius_m=50, rotation_pct=0.25, clockwise=False)
    wobbly_road = SineLaneSegment(period_m=50, amplitude_m=10, offset_m=12.5, num_cycles=2)

    lane = LaneDefinition([
        straight_segment,
        arc_left,
        straight_segment,
        wobbly_road,
        straight_segment,
        arc_left,
        straight_segment,
        arc_left,
        straight_segment,
        wobbly_road,
        straight_segment,
        arc_left
    ],
    num_lanes=4)

    lane.draw()

def test_lanes():
    test_straight_road()
    test_matt_track(True)
    test_matt_track(False)
    test_sine_wobbly_road()
    test_arc_wobbly_road()
    test_everything()

def test_actors_and_lanes():

    # make a matt track
    straight_segment = StraightLaneSegment(50)
    arc_segment = ArcLaneSegment(radius_m=50, rotation_pct=.5, clockwise=True)
    straight_segment = StraightLaneSegment(10)
    arc_left = ArcLaneSegment(radius_m=50, rotation_pct=0.25, clockwise=False)
    wobbly_road = SineLaneSegment(period_m=100, amplitude_m=10, offset_m=25, num_cycles=2)

    lane = LaneDefinition([
        straight_segment,
        arc_segment,
        straight_segment,
        arc_segment
    ],
    num_lanes=4)

    lane = LaneDefinition([
        straight_segment,
        arc_left,
        straight_segment,
        wobbly_road,
        straight_segment,
        arc_left,
        straight_segment,
        arc_left,
        straight_segment,
        wobbly_road,
        straight_segment,
        arc_left
    ],
    num_lanes=4)

    # make some actor that go along the lanes
    actors = []
    actors.append(PathFollowingActor('test', lane.lane_dict[0], 0, 0))
    actors.append(PathFollowingActor('test', lane.lane_dict[1], 0, 300))
    actors.append(PathFollowingActor('test', lane.lane_dict[2], 0, 600))
    actors.append(PathFollowingActor('test', lane.lane_dict[3], 0, 700))

    fig, ax = plt.subplots()
    lane.draw(fig, ax, show=False)
    for actor in actors:
        actor.draw(fig, ax, show=False)

    plt.show()


test_lanes()
test_actors_and_lanes()