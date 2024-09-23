#pragma once

#define _USE_MATH_DEFINES
#include "lyriq_model/CavLyriqModel.h"
#include "controller.h"
#include <Eigen/Dense>
#include <climits>
#include <cmath>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <iostream>

using namespace Eigen;
using namespace std;

typedef Matrix<double, Dynamic, 3> MatrixN3d;
typedef Matrix<double, 1, 3> Matrix13d;

// path to project base
inline const std::string PROJECT_BASE = "../";

/**
* @brief Returns 2D rotation matrix corresponding to heading angle parameter
* @param heading_rad  rotation angle in radians
* @return rotMat 2D rotation matrix
*/
Eigen::Matrix2d getRotMat(double heading_rad);

/**
The Path2D class implements a simple 2D path using waypoints that specify x (forward positive), y (left positive), and headings (counterclockwise positive). See LaneDefinition and PathFollowingActor classes for examples of usage.
*/
class Path2D {
public:

    /** @brief Nx3 matrix containing waypoints of path in the form [x_m, y_m, heading_rad] */
    MatrixN3d waypoints;

    /** @brief Nx1 vector containing distances along the path to each waypoint */
    VectorXd distances;

    /** @brief explicit definition of base constructor */
    Path2D();

    /**
    @brief Constructor for Path2D. Takes in an Nx3 matrix of waypoints in the form [x_m, y_m, heading_rad]
    @param waypoints shared pointer to Eigen waypoints matrix
    */
    Path2D(MatrixN3d& waypoints);

    /** @brief updates distances with the contents in waypoints matrix and regenerates interpolators */
    void updateDistance();

    /**
    @brief interpolates a waypoint based on the provided distance along the path
    @param distance_m the "arc length distance" along the 2D path
    @return 1x3 Eigen matrix in the form [x_m, y_m, heading_rad]
    */
    Matrix13d distanceToWaypoint(double distance_m);

    /**
    @brief transforms a path by rotating it by specified heading, then translating by specified x and y
    @param x_m x transformation distance (m)
    @param y_m y transformation distance (m)
    @param heading_rad rotation angle (rad)
    */
    void transformPath(double x_m = 0, double y_m = 0, double heading_rad = 0);

    /**
    @brief transforms a path by taking the heading vector of each waypoint and shifting the path perpendicularly. This distorts the path, but preserves its overall shape, so it can be used for stacking paths laterally without overlap.
    @param offset_m distance along heading vectors to transform path (m)
    @param left boolean value indicating whether transformation is to left or right
    @return pointer to current object to allow for operator chaining
    */
    Path2D* transformPathAlongHeading(double offset_m, bool left = true);

    /**
    @brief operator overloading of + to perform path concatenation (Path2D p = p1 + p2)
    @param p2 const reference to Path2D to concatenate to this Path2D
    @return concatenated path
    */
    Path2D operator+ (Path2D const &p2);

    /** 
    @brief returns x values of waypoints
    @return VectorXd of x values
    */
    VectorXd getXVals();
    
    /** 
    @brief returns y values of waypoints
    @return VectorXd of y values
    */
    VectorXd getYVals();

    /** 
    @brief returns heading values of waypoints
    @return VectorXd of heading values
    */
    VectorXd getHeadings();

    void print(int startIdx = 0, int endIdx = INT_MAX);

};


/**
Base class to store common data/logic for lane segments
*/
class LaneSegment {
    public:

    /** @brief Path2D defining lane waypoints */
    Path2D path;

    /** @brief explicit definition of default constructor */
    LaneSegment();

    /**
    @brief base LaneSegment constructor
    @param waypoints shared pointer to waypoints matrix 
    */
    LaneSegment(MatrixX3d& waypoints);

};

/**
Inherited class for creating straight line segments
*/
class StraightLaneSegment : public LaneSegment {
    public:

    /** @brief length of lane segment in meters */
    double length_m;

    /** @brief resolution of lane segment waypoints in meters */
    double resolution_m;

    /**
    @brief constructor for a straight lane -- fully definable by a length
    @param length_m length of straight line segment in meters
    @param resolution_m spacing between waypoints in meters, 0.1 by default
    */
    StraightLaneSegment(double length_m, double resolution_m = 0.1);

};

/**
Inherited class for creating arc line segments
*/
class ArcLaneSegment : public LaneSegment {
    public:

    /** @brief turn radius of arced lane segment */
    double radius_m;

    /** @brief rotation percentage of arced lane segment */
    double rotation_pct;

    /** @brief is arced segment clockwise or counterclockwise? */
    bool clockwise;

    /** @brief resolution of lane segment waypoints in meters (pre-transformation) */
    double resolution_m;

    /**
    @brief constructor for an arced lane -- fully definable by a turn radius, percentage of rotation, and CW/CCW
    @param radius_m length of turn radius in meters
    @param rotation_pct percentage of a full rotation (between 0 and 1) -- eg. half turn = 0.5
    @param clockwise boolean defining if lane arcs CW or CCW
    @param resolution_m spacing between waypoints (pre-transformation) in meters, 0.001 by default
    */
    ArcLaneSegment(double radius_m, double rotation_pct, bool clockwise, double resolution_m = 0.001);

};

/**
Inherited class for creating sine line segments
*/
class SineLaneSegment : public LaneSegment {
    public:

    /** @brief period of sine wave in meters */
    double period_m;

    /** @brief amplitude of sine wave in meters */
    double amplitude_m;

    /** @brief offset of sine wave in meters (period * phaseshift / 2pi) */
    double offset_m;

    /** @brief number of cycles for sine wave segment */
    double numCycles;

    /** @brief resolution of lane segment waypoints in meters (pre-transformation) */
    double resolution_m;

    /**
    @brief constructor for a sine lane -- fully definable by period, amplitude, an offset, and a number of cycles
    @param period_m period of sine wave meters
    @param amplitude_m amplitude of sine wave in meters
    @param offset_m offset for sine wave (meters), equal to (phase shift * period) / 2pi
    @param numCycles number of cycles for sine wave segment
    @param resolution_m spacing between waypoints (pre-transformation) in meters, 0.001 by default
    */
    SineLaneSegment(double period_m, double amplitude_m, double offset_m, double numCycles, double resolution_m = 0.001);

};

/**
Class fully defining a lane as a set of segments in a particular order with a number of lanes and a specified lane width 
*/
class LaneDefinition {
    public:

    /** @brief map of lane id to path defining centerlane of full lane (lane id 0 = rightmost lane, increasing id to the left) */
    unordered_map<int, Path2D> laneMap;

    /** @brief map of lane boundary id to path defining full lane boundary path (same numbering rules as centerlanes) */
    unordered_map<int, Path2D> laneBoundaryMap;

    /** @brief number of lanes in lane definition */
    int numLanes;

    /** @brief lane width in meters */
    double laneWidth_m;

    LaneDefinition();

    /**
    @brief constructor for a lane definition, defined by a list of lane segments defining the full path of the rightmost lane, a number of lanes, and a lane width
    @param segments vector of LaneSegment objects (can be children as well like StraightLaneSegment, etc.)
    @param numLanes number of lanes the lane definition defines (additional lanes will be projected to the left of the first lane)
    @param laneWidth_m width of lanes (currently the same width for all lanes)
    */
    LaneDefinition(vector<LaneSegment>& segments, int numLanes = 1, double laneWidth_m = 3.5);

    /**
    @brief get waypoints of a specified lane in the world space
    @param laneId id of centerlane waypoints to retrieve (0 = rightmost lane)
    @return Nx3 eigen matrix of centerlane waypoints 
    */
    Path2D getLaneWaypoints(int laneId);

    /**
    @brief get waypoints of a specified lane in the world space
    @param laneBoundaryId id of centerlane waypoints to retrieve (0 = rightmost lane boundary)
    @return Nx3 eigen matrix of lane boundary waypoints 
    */
    Path2D getLaneBoundaryWaypoints(int laneBoundaryId);

};

/**
Base Actor class defining dimensions and state in the environment
*/
class Actor {
public:

    /** @brief actor name */
    string name;

    /** @brief length of actor in meters */
    double length_m;

    /** @brief width of actor in meters */
    double width_m;

    /** @brief state of geometric center of actor Eigen row vector of [x, y, heading] */
    Matrix13d state;

    /** @brief explicit definition of default constructor */
    Actor();

    /**
    @brief constructor for base actor class
    @param name name of actor
    @param length_m length of actor
    @param width_m width of actor
    @param init_state Eigen row vector of [x, y, heading] initial actor state (deafults to [0,0,0]) 
    */
    Actor(string name, double length_m, double width_m, Matrix13d init_state = Matrix13d({{0,0,0}}));

    double getX();
    double getY();
    double getHeading();

    //virtual void update(double dt) { std::cout << "this should not be called\n"; };
    virtual void update(double dt) { std::cout << "this should not be called\n"; };

};

class DriveCycle {

public:

    vector<double> times, speeds_mps;

    DriveCycle();
    DriveCycle(string csvPath);
    DriveCycle(vector<double>& times, vector<double>& speeds);

    double getSpeed(double time);

};

/**
Child of Actor class that gives actor a path to follow and a speed along said path
*/
class PathFollowingActor : public Actor {
public:

    /** @brief path for vehicle to follow */
    Path2D path;

    /** @brief speed of actor in m/s */
    double speed_mps;

    /** @brief distance along path in m */
    double distAlongPath;

    shared_ptr<DriveCycle> driveCycle;

    double time;

    /**
    @brief constructor for PathFollowingActor
    @param name name of actor
    @param path Path2D representing path for vehicle to follow
    @param length_m length of actor (default = 4m)
    @param width_m width of actor (default = 2m)
    @param initDist_m initial distance of actor along path (default = 0)
    @param speed_mps initial speed in m/s (default = 0)
    */
    PathFollowingActor(
        string name, 
        Path2D path, 
        double speed_mps = 0, 
        double initDist_m = 0, 
        double length_m = 4, 
        double width_m = 2,
        shared_ptr<DriveCycle> driveCycle = nullptr
    );

    /**
    @brief update actor's position on path after a time delta dt
    @param dt timedelta in seconds
    */
    void update(double dt) override;

};

class EgoVehicle : public Actor {
public:

    bool cavEnable = 0;
    int gearCmd = 0; // 1 = drive, 0 = neutral, -1 = reverse
    double accelPedalPos_pct = 0, brakePedalPos_pct = 0, steeringWheelAngle_rad = 0; // +ve = CCW
    double cavAxlTrqReq_Nm = 0, cavBrakeTrqReq_mps2 = 0, cavSteeringTrqReq_Nm = 0;
    int timestep = 0;
    Matrix13d init_state;
    
    CavLyriqModel model;

    /** @brief explicit definition of default constructor */
    EgoVehicle (
        string name = "ego",
        double init_x = 0,
        double init_y = 0,
        double init_heading = 0,
        double length_m = 4, 
        double width_m = 2
    );

    /**
    @brief update ego's position on path after a time delta dt
    @param dt timedelta in seconds
    */
    void update(double dt) override;

    // simple getters implemented in header file for simplicity
    double getVx_mps() { return this->model.getExternalOutputs().VehFdbk.Body.BdyFrm.Cg.Vel.xdot; };
    double getAx_mps2() { return this->model.getExternalOutputs().VehFdbk.Body.BdyFrm.Cg.Acc.xddot; };
    double getYawRate_radps() { return this->model.getExternalOutputs().VehFdbk.Body.BdyFrm.Cg.AngVel.r; };
    double getSOC_pct() { return this->model.getExternalOutputs().VehFdbk.Battery.BattSoc; }

    void reset();

};

class SimController {

public:

    double sampleRate_Hz;
    double lastExecutedTime;

    SimController(double sampleRate_Hz = 20);
    virtual void init() = 0;
    virtual void step(double ts) = 0;

};

class BeseeCore {
public:

    int updateRateHz;
    vector<shared_ptr<PathFollowingActor>> actors;
    LaneDefinition lanes;
    EgoVehicle ego;
    SimController* simController;

    /** @brief explicit definition of default constructor */
    BeseeCore();

    /** 
    @brief constructor for initializing besee with a scenario definition from scenarios.yaml 
    @param scenarioName name of scenario to select from scenarios.yaml
    @param updateRateHz frequency at which to step besee in Hz (default = 100Hz)
    */
    BeseeCore(string scenarioName, int updateRateHz = 100);

    /**
    @brief constructor for initializing besee with appropriate objects (ego vehicle, actors, lanes)
    @param ego reference to EgoVehicle object
    @param actors reference to vector of shared pointers to Actor objects (needs to be pointers to handle polymorphism)
    @param lanes reference of LaneDefinition object defining road structure
    */
    BeseeCore(
        EgoVehicle& ego,
        vector<shared_ptr<PathFollowingActor>>& actors, 
        LaneDefinition& lanes,
        int updateRateHz = 100
    );

    /**
    @brief update besee's actors and ego vehicle
    @param dt timedelta in seconds
    */
    void update(double dt);

    /**
    @brief handle all signal logging with loggers library here
    @param dt timestep to log
    */
    void logSignals(int timestep);

    /**
    @brief run besee for a specified time and time ratio
    @param endTime time in seconds to run scenario
    */
    void runSim(double endTime = -1, double timeRatio = 0);

    /**
     * @brief get actor by name
     * @param string represnting actor's name
     * @return shared pointer to actor if it exists. nullptr otherwise.
     */
    shared_ptr<PathFollowingActor> getActor(std::string actorName);

    /**
    @brief get actor ground truth states in the ego vehicle frame as a map of actor name to vector state in [x, y, vx, vy, heading]
    @return reference to map from actor names to eigen vector representing actor ground truth states
    */
    unordered_map<string, VectorXd>& getActorStatesInEgoFrame();

    /**
    @brief get cubic polynomial approximations of all lanes boundaries in the ego frame
    @return reference to matrix representing polynomial coefficients for all lane boundaries where each row is in the form [c3, c2, c1, c0] (y = c3*x^3 + c2*x^2 + c1*x + c0) 
    */
    MatrixX4d* getLanePolynomialsInEgoFrame();

    /**
    @brief set cav longitudinal commands (torque request, brake decel req) -- note besee will hold these commands until the function is called again 
    @param propTorqueReq_Nm propulsive torque command in Nm
    @param brakeReq_mps2 brake request in m/s^2
    */
    void setCavLongitudinalCommands(double propTorqueReq_Nm, double brakeReq_mps2);

    /**
    @brief set cav lateral command (steering torque request) -- note besee will hold these commands until the function is called again 
    @param steeringTorqueReq_Nm steering torque request in Nm
    */
    void setCavLateralCommands(double steeringTorqueReq_Nm);

    /**
    @brief manually set accel and brake pedal commands -- note besee will hold these commands until the function is called again
    @param accelPedal_pct accelerator pedal position as a percentage [0, 1]
    @param brakePedal_pct brake pedal position as a percentage [0, 1]
    */
    void setManualPedalCommands(double accelPedal_pct, double brakePedal_pct);

    /**
    @brief manually set steering wheel angle command -- note besee will hold these commands until the function is called again
    @param steeringWheelPosition_rad steering wheel position in radians (left = negative, right = positive)
    */
    void setSteeringWheelPosition(double steeringWheelPosition_rad);

    /**
    @brief set toggle for if cav commands are enabled -- note manual brake command will override if it is set
    @param cavEnable boolean representing if cav commands will be acknowledged
    */
    void setCavEnable(bool cavEnable);

    // deprecated : besee viz rewritten in python with matplotlib
    void drawInit();
    void drawUpdate(int ts);

};

class SimCaccController : public SimController {

public:

    CaccController controller;
    BeseeCore* env;
    double setSpeed;

    SimCaccController(BeseeCore* env, double setSpeed, double sampleRate_Hz = 50);
    void init() override;
    void step(double ts) override;

};