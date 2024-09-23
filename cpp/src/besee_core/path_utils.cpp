#include "besee_utils.h"
#include "matrix_utils.h"
#include <cstddef>
#include <algorithm>

Path2D::Path2D() { 
    this->waypoints = MatrixN3d(1, 3); 
    this->waypoints(0, 0) = 0;
    this->waypoints(0, 1) = 0;
    this->waypoints(0, 2) = 0;
}

Path2D::Path2D(MatrixN3d& waypoints) {

    this->waypoints = waypoints;
    this->updateDistance();

}

void Path2D::updateDistance() {

    // get first differences of x and y vectors from waypoints
    VectorXd x_vals = this->getXVals();
    VectorXd x_deltas = getFirstDifferences(x_vals);

    VectorXd y_vals = this->getYVals();
    VectorXd y_deltas = getFirstDifferences(y_vals);

    // calculate norm of delta vectors to get distances between each waypoint
    VectorXd distances = sqrt((x_deltas.array() * x_deltas.array() + y_deltas.array() * y_deltas.array()).array());
    
    // set class attribute
    this->distances = cumsum(distances);

}

Matrix13d Path2D::distanceToWaypoint(double distance_m) {

    VectorXd x_vals = (this->waypoints)(all, 0);
    VectorXd y_vals = (this->waypoints)(all, 1);
    VectorXd headings = (this->waypoints)(all, 2);

    Matrix13d waypoint(
        {{interpolate1d(this->distances, x_vals, distance_m),
        interpolate1d(this->distances, y_vals, distance_m),
        interpolate1d(this->distances, headings, distance_m)}}
    );

    return waypoint;

}

void Path2D::transformPath(double x_m, double y_m, double heading_rad) {

    // rotate path if heading is nonzero
    if (heading_rad != 0) {
        // get rotation matrix for each waypoint about the origin
        Matrix2d rotMat = getRotMat(-heading_rad);

        // apply rotation matrix to each row of the array and set waypoints arrray
        MatrixX2d xy_vals = (this->waypoints).leftCols(2);
        for (int i = 0; i < (this->waypoints).rows(); i++) {
            xy_vals.row(i) *= rotMat;
        }
        (this->waypoints).leftCols(2) = xy_vals;

        // add heading to current heading vector
        (this->waypoints).col(2).array() += heading_rad;
    }

    // translate path by x if x is nonzero
    if (x_m != 0) { (this->waypoints).col(0).array() += x_m; }

    // translate path by y if y is nonzero
    if (y_m != 0) { (this->waypoints).col(1).array() += y_m; }

    // update distance
    this->updateDistance();

}

Path2D* Path2D::transformPathAlongHeading(double offset_m, bool left) {

    // get waypoint headings of centerlane and construct vector of specified length in that direction
    VectorXd laneOffsetsX = offset_m * (cos(this->getHeadings().array()));
    VectorXd laneOffsetsY = offset_m * (sin(this->getHeadings().array()));
    
    // Nx2 matrix where each row is heading vector of magnitude offset
    MatrixX2d laneOffsetVectors(laneOffsetsX.rows(), 2);
    laneOffsetVectors << laneOffsetsX, laneOffsetsY; 

    // if left, rotate vectors by -pi/2. if right, rotate by pi/2
    int sign = left ? -1 : 1;
    Matrix2d rotMat = getRotMat(sign * M_PI/2);
    for (int i = 0; i < laneOffsetVectors.rows(); i++) {
        laneOffsetVectors.row(i) *= rotMat;
    }

    // add a heading_deltas column of 0s
    MatrixX3d waypoint_deltas(laneOffsetVectors.rows(), 3);
    waypoint_deltas << laneOffsetVectors, VectorXd::Zero(laneOffsetVectors.rows());

    // add waypoint deltas to waypoints and update distance
    (this->waypoints) += waypoint_deltas;
    this->updateDistance();

    // return reference to itself to allow for operator chaining
    return this;

}

Path2D Path2D::operator+(Path2D const& p2) {

    // need to copy this to ensure original path does not get mutated...
    // TODO: see if the performance hit of copying vector is significant.
    Path2D temp_path = Path2D(p2);

    // transform input path to be continuous with end of existing path
    int lastIndex = (this->waypoints).rows() - 1;
    double init_x = (this->waypoints)(lastIndex,0);
    double init_y = (this->waypoints)(lastIndex,1);
    double init_heading = (this->waypoints)(lastIndex,2);
    temp_path.transformPath(init_x, init_y, init_heading);

    // vertically concatente this path's waypoints with the incoming transformed waypoints 
    int numCombinedWaypoints = this->waypoints.rows() + temp_path.waypoints.rows();
    MatrixX3d newWaypoints(numCombinedWaypoints, 3);
    newWaypoints << (this->waypoints), (temp_path.waypoints);

    // set new waypoints and update distance
    (this->waypoints) = newWaypoints;
    this->updateDistance();

    return *this;

}

VectorXd Path2D::getXVals() { return (VectorXd)((this->waypoints).col(0)); }
VectorXd Path2D::getYVals() { return (VectorXd)((this->waypoints).col(1)); }
VectorXd Path2D::getHeadings() { return (VectorXd)((this->waypoints).col(2)); }

void Path2D::print(int startIdx, int endIdx) {

    for (int i = startIdx; i < min((int)(waypoints).rows(), endIdx); i++) {
        cout << (waypoints)(i, 0) << ", " << (waypoints)(i, 1) << ", " << (waypoints)(i, 2) << "\n";
    }

}