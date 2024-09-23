#include <besee_utils.h>
#include <matrix_utils.h>
#include <memory>
#include <unordered_map>
#include <utility>

// default constructor for LaneSegment
LaneSegment::LaneSegment() : path(Path2D()) {}

// waypoint constructor for LaneSegment
LaneSegment::LaneSegment(MatrixX3d& waypoints) : path(Path2D(waypoints)) {}

StraightLaneSegment::StraightLaneSegment(double length_m, double resolution_m) : 
    length_m(length_m),
    resolution_m(resolution_m) 
{

    // for straight line, x values are just equally spaced from 0 to length, y values and headings are all 0
    int numWaypoints = (int)(length_m / resolution_m);
    VectorXd xVals = VectorXd::LinSpaced( numWaypoints, 0, length_m);
    VectorXd yVals = VectorXd::Zero(numWaypoints);
    VectorXd headings = VectorXd::Zero(numWaypoints);

    // construct waypoints array, create path, set as path of parent class
    MatrixX3d waypoints = MatrixX3d(numWaypoints, 3);
    waypoints << xVals, yVals, headings;
    this->path = Path2D(waypoints);

}

ArcLaneSegment::ArcLaneSegment(double radius_m, double rotation_pct, bool clockwise, double resolution_m) :
    radius_m(radius_m),
    rotation_pct(rotation_pct),
    clockwise(clockwise),
    resolution_m(resolution_m)
{

    // for arc line, generate angle vector theta [0, 2pi*rotation_pct]
    // x values are r*sin(theta), y values are r - r*cos(theta), headings are theta.
    // if arc is CW, y values and headings are multiplied by -1
    int numWaypoints = (int)(2.0 * M_PI * this->rotation_pct / this->resolution_m);
    int sign = clockwise ? -1 : 1;
    VectorXd thetas = VectorXd::LinSpaced(numWaypoints, 0, 2*M_PI*this->rotation_pct);
    VectorXd xVals = this->radius_m * sin(thetas.array());
    VectorXd yVals = sign * (this->radius_m - this->radius_m*cos(thetas.array()));
    VectorXd headings = sign * thetas;

    // construct waypoints array, create path, set as path of parent class
    auto waypoints = MatrixX3d(numWaypoints, 3);
    waypoints << xVals, yVals, headings;
    this->path =  Path2D(waypoints);

}

SineLaneSegment::SineLaneSegment(double period_m, double amplitude_m, double offset_m, double numCycles, double resolution_m) :
    period_m(period_m),
    amplitude_m(amplitude_m),
    offset_m(offset_m),
    resolution_m(resolution_m)
{

    // for sine lane segments, x values are just 0 to period*numcycles, y values are a*sin(2pi/T (x - d)) - y0, headings are arctan(dy/dx)
    int numWaypoints = (this->numCycles * this->period_m) / this->resolution_m;
    VectorXd xVals = VectorXd::LinSpaced(numWaypoints, 0, this->numCycles * this->period_m);
    VectorXd yVals = this->amplitude_m * sin(2 * M_PI / this->period_m * (xVals.array() - this->offset_m));
    yVals.array() -= yVals(0);
    auto headings = atan( 2 * M_PI * this->amplitude_m / this->period_m * cos( 2 * M_PI / this->period_m * (xVals.array() - this->offset_m) ) );

    // construct waypoints array, create path, set as path of parent class
    auto waypoints = MatrixX3d();
    waypoints << xVals, yVals, headings;
    this->path =  Path2D(waypoints);
    
}

LaneDefinition::LaneDefinition() {}

LaneDefinition::LaneDefinition(vector<LaneSegment>& segments, int numLanes, double laneWidth_m) :
    numLanes(numLanes),
    laneWidth_m(laneWidth_m)
{

    // create initial waypoint at [0, 0, 0] construct full path from lane segments
    MatrixX3d zeroWaypoint = MatrixX3d(MatrixX3d({{0, 0, 0}}));
    Path2D basePath(zeroWaypoint);
    for (LaneSegment segment : segments) {
        basePath = basePath + segment.path;
    }

    // set initial right lane boundary (id 0)
    this->laneBoundaryMap.insert(make_pair(0, *Path2D(basePath).transformPathAlongHeading(this->laneWidth_m/2, false) ));

    // create numLanes new lanes/lane boundaries, offset them by the approproate amount, insert them into maps
    for (int i = 0; i < this->numLanes; i++) {

        this->laneMap.insert(make_pair(i, *Path2D(basePath).transformPathAlongHeading(i*this->laneWidth_m) ));
        this->laneBoundaryMap.insert(make_pair(i + 1, *Path2D(basePath).transformPathAlongHeading(i*this->laneWidth_m + this->laneWidth_m/2)) );

    }

}

Path2D LaneDefinition::getLaneBoundaryWaypoints(int laneBoundaryId) {
    return this->laneBoundaryMap.at(laneBoundaryId);
}

Path2D LaneDefinition::getLaneWaypoints(int laneId) {
    return this->laneMap.at(laneId);
}
