#include <besee_utils.h>
#include <chrono>
#include <climits>
#include <memory>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

#include "utils/logging/loggers.h"

using namespace std::chrono;

long getEpochTimeUsec() { return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count(); }

// for debugging purposes if i don't know what a node structure is...
void walkNode(YAML::Node node) {
    switch (node.Type()) {
    case YAML::NodeType::Null: // ...
    case YAML::NodeType::Scalar: // ...
        std::cout << node.as<string>() << "\n";
        break;
    case YAML::NodeType::Sequence:
        for (auto n : node) {
        walkNode(n);
        // recurse on "element"
        }
        break;
    case YAML::NodeType::Map:
        for (auto p : node) {
        auto key = p.first;
        std::cout << key.as<string>() << "\n";
        auto value = p.second;
        walkNode(key);
        walkNode(value);
        // recurse on "key" and "value"
        // if you're sure that "key" is a string, just grab it here
        }
        break;
    case YAML::NodeType::Undefined: // ...
    break;
    }
}

DriveCycle* parseDriveCycleCsv(string shortFilename) {
    string fullPath = PROJECT_BASE + "config/drive_cycle/" + shortFilename + ".csv";
    return new DriveCycle(fullPath);
}

BeseeCore::BeseeCore() {}

BeseeCore::BeseeCore(EgoVehicle& ego, vector<shared_ptr<PathFollowingActor>>& actors, LaneDefinition& lanes, int updateRateHz) :
    actors(actors),
    lanes(lanes),
    updateRateHz(updateRateHz) 
{
    this->ego = ego;
}

BeseeCore::BeseeCore(string scenarioName, int updateRateHz) {

    // load yaml configuration files -- scenarios, actors, roads
    YAML::Node allScenarios = YAML::LoadFile(PROJECT_BASE + "config/scenarios.yaml");
    YAML::Node allActors = YAML::LoadFile(PROJECT_BASE + "config/actors.yaml");
    YAML::Node allRoads = YAML::LoadFile(PROJECT_BASE + "config/roads.yaml");

    // load scenario YAML Node, get roads, actors, ego vehicle, controller from other config files
    auto scenarioNode = allScenarios[scenarioName];
    auto roadsNode = allRoads[scenarioNode["lanes"].as<string>()];
    auto actorsNode = scenarioNode["actors"];
    auto egoNode = scenarioNode["ego"];
    string controllerName = scenarioNode["controller"].as<string>();

    // construct vector of lane segments by iterating through list of yaml lane dicts
    vector<LaneSegment> laneSegments;
    for (auto segment : roadsNode["lane_segments"]) {
        
        for (auto map : segment) {

            string laneType = map.first.as<string>();
            auto laneProperties = map.second;

            // if dict key is "straight", parse StraightLaneSegment
            if (laneType == "straight") {
                laneSegments.push_back(
                    StraightLaneSegment(laneProperties["length"].as<double>())
                );
            }

            // if dict key is "arc", parse ArcLaneSegment
            else if (laneType == "arc") {
                laneSegments.push_back(
                    ArcLaneSegment(
                        laneProperties["radius"].as<double>(), 
                        laneProperties["rotations"].as<double>(),
                        laneProperties["clockwise"].as<bool>()
                    )
                );
            }

        }
    }

    // construct LaneDefinition object
    LaneDefinition lanes = LaneDefinition(
        laneSegments,
        roadsNode["num_lanes"].as<int>(),
        roadsNode["lane_width"].as<double>()
    );

    // construct vector of shared pointers of actors
    vector<shared_ptr<PathFollowingActor>> actors;

    for (auto actor : actorsNode) {

        for (auto map : actor) {

            string vehName = map.first.as<string>();
            auto vehProperties = map.second; 
            string vehModel = vehProperties["make"].as<string>();

            shared_ptr<DriveCycle> driveCycle = vehProperties["speed_profile"] ? 
                shared_ptr<DriveCycle>(parseDriveCycleCsv(vehProperties["speed_profile"].as<string>())) : 
                nullptr;

            actors.push_back(make_shared<PathFollowingActor>(
                vehName,
                lanes.getLaneWaypoints(vehProperties["lane_id"].as<int>()),
                vehProperties["speed"] ? vehProperties["speed"].as<double>() : 0,
                vehProperties["offset"] ? vehProperties["offset"].as<double>() : 0,
                allActors[vehModel]["length"].as<double>(),
                allActors[vehModel]["width"].as<double>(),
                driveCycle
            ));

        }

    }

    // construct ego vehicle
    string egoMake = egoNode["make"].as<string>();
    EgoVehicle ego(
        "ego",
        0,
        0,
        0,
        allActors[egoMake]["length"].as<double>(),
        allActors[egoMake]["width"].as<double>()
    );

    // set sim controller 
    if (controllerName == "cacc") {
        this->simController = new SimCaccController(this, 27.777);
    }
    else {
        cout << "NOTE: CONTROLLER TYPE " << controllerName << " NOT SUPPORTED YET. DEFAULTING TO CACC." << "\n";
        this->simController = new SimCaccController(this, 27.777);
    }

    // set member variables
    this->actors = actors;
    this->lanes = lanes;
    this->updateRateHz = updateRateHz;
    this->ego = ego;

}

void BeseeCore::update(double dt) {
    ego.update(1.0f / updateRateHz);
    for (auto actor : this->actors) {actor->update(1.0f / updateRateHz); }
}

void BeseeCore::runSim(double endTime, double timeRatio) {

    // set update rate for simulation
    double dt = 1.0/this->updateRateHz;

    // initialize controller
    this->simController->init();

    // if endTime param is -1, run indefinitely
    if (endTime == -1) { endTime = INT_MAX; }

    // main simulation loop
    for (int ts = 0; ts < (endTime*this->updateRateHz); ts++) {

        // get start clock time
        long startClockTime = getEpochTimeUsec();

        // run controller step
        this->simController->step((double)ts / (double)this->updateRateHz);

        // update actors in world 
        this->update(dt);

        // printing stuff
        std::cout << "=========== TS = " << (double)ts/(double)this->updateRateHz << "sec ==========" << "\n";
        std::cout << "Ego Pos: " << this->ego.getX() << ", " << this->ego.getY() << "\n";
        for (auto actor : this->actors) {
            cout << actor->name << " Pos: " << actor->getX() << ", " << actor->getY() << "\n";
        }

        // save logging signals
        this->logSignals(ts);

        // get end clock time and do not continue until the next timestep is ready to go
        while ((getEpochTimeUsec() - startClockTime) < (dt * timeRatio) * 1.0e6) {}

    }

    // sim terminate
    logger100Hz.writeLogFileCsv();

}

void BeseeCore::logSignals(int timestep) {

    // log ego signals (x, y, heading)
    LOG_SIGNAL_TS("ACTOR_ego_x", timestep, ego.getX());
    LOG_SIGNAL_TS("ACTOR_ego_y", timestep, ego.getY());
    LOG_SIGNAL_TS("ACTOR_ego_heading", timestep, ego.getHeading());
    LOG_SIGNAL_TS("ego_soc", timestep, ego.getSOC_pct());
    LOG_SIGNAL_TS("ego_speed", timestep, ego.getVx_mps());
    LOG_SIGNAL_TS("ego_acceleration", timestep, ego.getAx_mps2());
    LOG_SIGNAL_TS("pedal_pos", timestep, ego.accelPedalPos_pct);
    LOG_SIGNAL_TS("brake_pos", timestep, ego.brakePedalPos_pct);
    LOG_SIGNAL_TS("cav_enable", timestep, ego.cavEnable);
    LOG_SIGNAL_TS("cav_torque_req", timestep, ego.cavAxlTrqReq_Nm);
    LOG_SIGNAL_TS("cav_brake_req", timestep, ego.cavBrakeTrqReq_mps2);
    LOG_SIGNAL_TS("cav_steering_torque_req", timestep, ego.cavSteeringTrqReq_Nm);

    // log actor states (x, y, heading)
    for (auto actor : actors) {
        LOG_SIGNAL_TS("ACTOR_" + actor->name + "_x", timestep, actor->getX());
        LOG_SIGNAL_TS("ACTOR_" + actor->name + "_y", timestep, actor->getY());
        LOG_SIGNAL_TS("ACTOR_" + actor->name + "_heading", timestep, actor->getHeading());
        LOG_SIGNAL_TS("ACTOR_" + actor->name + "_speed", timestep, actor->speed_mps);
    }

}

shared_ptr<PathFollowingActor> BeseeCore::getActor(std::string actorName) {
    /* TODO: replace implementation of actor vector with map to avoid iterating through everything */
    shared_ptr<PathFollowingActor> returnActor = nullptr;
    for (auto actor : actors) {
        if (actor->name == actorName) {
            returnActor = actor;
            break;
        }
    }
    return returnActor;
}

unordered_map<string, VectorXd>& BeseeCore::getActorStatesInEgoFrame() {

    // initialize actor state map
    unordered_map<string, VectorXd>* actorStates = new unordered_map<string, VectorXd>();

    // get ego vehicle state in world frame
    Matrix13d egoState = this->ego.state;
    double egoVel = this->ego.getVx_mps();

    // iterate through all actors
    for (shared_ptr<PathFollowingActor> actor : this->actors) {

        // get actor position, heading, and speed
        Matrix13d actorState = actor->state; // [x, y, heading] in world frame
        double actorSpeed = actor->speed_mps;

        // subtract ego state from actor state to get relative state
        auto relState = actorState.array() - egoState.array();
        double deltaHeading = relState(0, 2);

        // get rotation matrix and rotate relative state by delta heading to get proper x and y values
        auto rotMat = getRotMat(-deltaHeading);
        MatrixXd relXyVals = relState.leftCols(2);
        relXyVals *= rotMat;

        // get components of speed vector in x and y directions in ego frame
        double leadSpeedX = actorSpeed * cos(deltaHeading);
        double leadSpeedY = actorSpeed * sin(deltaHeading);

        // subtract ego vehicle speed from x speed (currently assuming vehicle has no y speed.)
        // TODO: If necessary, implement ego y speed
        leadSpeedX -= egoVel;

        // create actor state vector and add to actor state map
        VectorXd relActorState(5, 1);
        relActorState << relXyVals(0), relXyVals(1), leadSpeedX, leadSpeedY, deltaHeading;
        actorStates->insert(
            pair(actor->name, relActorState)
        );

    }

    return *actorStates;

}

MatrixX4d* BeseeCore::getLanePolynomialsInEgoFrame() {
    // TODO
    return new Eigen::MatrixX4d();
}

void BeseeCore::setCavLongitudinalCommands(double propTorqueReq_Nm, double brakeReq_mps2) {
    this->ego.cavAxlTrqReq_Nm = propTorqueReq_Nm;
    this->ego.cavBrakeTrqReq_mps2 = brakeReq_mps2;    
}

void BeseeCore::setCavLateralCommands(double steeringTorqueReq_Nm) { this->ego.cavSteeringTrqReq_Nm = steeringTorqueReq_Nm; }

void BeseeCore::setManualPedalCommands(double accelPedal_pct, double brakePedal_pct) {
    this->ego.accelPedalPos_pct = accelPedal_pct;    
    this->ego.brakePedalPos_pct = brakePedal_pct;    
}

void BeseeCore::setSteeringWheelPosition(double steeringWheelPosition_rad) { this->ego.steeringWheelAngle_rad = steeringWheelPosition_rad; }

void BeseeCore::setCavEnable(bool cavEnable) { this->ego.cavEnable = cavEnable; }