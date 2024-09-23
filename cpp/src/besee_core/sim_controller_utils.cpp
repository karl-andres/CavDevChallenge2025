#include "besee_utils.h"
#include "utils/logging/loggers.h"

SimController::SimController(double sampleRate_Hz) {
    this->sampleRate_Hz = sampleRate_Hz;
    this->lastExecutedTime = 0;
}

SimCaccController::SimCaccController(BeseeCore* env, double setSpeed, double sampleRate_Hz) {
    this->env = env;
    this->setSpeed = setSpeed;
    this->sampleRate_Hz = sampleRate_Hz;
    this->lastExecutedTime = 0;
}

void SimCaccController::init() {
    //this->controller.initCalibrations("default", PROJECT_BASE + "/cav_autera_software/calibrations/cacc_params.yaml");
}

void SimCaccController::step(double ts) {

    // early return if controller should not be run for this timestep
    if (ts < (this->lastExecutedTime + (1 / this->sampleRate_Hz))) { return; }
    this->lastExecutedTime = ts;

    // get vehicle speed
    double egoSpeed_mps = this->env->ego.getVx_mps();

    // default values
    bool leadExists = false;
    double leadXPos = 0;
    double leadXVel = 0;
    double leadYPos = 0;
    double leadYVel = 0;
    // Track* leadTrack = new Track();

    // get lead vehicle information in ego frame (if it exists)
    auto actors = this->env->getActorStatesInEgoFrame();
    auto leadVehicleIterator = actors.find("lead");

    // if lead exists, set lead speed and distance parameters
    if (leadVehicleIterator != actors.end()) {

        // get lead properties and calculate relative state from ego vehicle
        auto lead = this->env->getActor("lead");
        auto relLeadState = leadVehicleIterator->second; // x, vx, y, vy, heading

        // construct full state vector and dummy covariance matrix
        Eigen::VectorXd fullState(9);
        fullState << relLeadState[0], relLeadState[1], 0, relLeadState[2], relLeadState[3], 0, 0, 0, 0;
        Eigen::MatrixXd covar = Eigen::MatrixXd::Zero(9, 9);

        leadXPos = relLeadState[0];
        leadXVel = relLeadState[1];
        leadYPos = relLeadState[2];
        leadYVel = relLeadState[3];
        leadExists = true;
    }

    double torqueCommand_nm;
    double brakeCommand_mps2;
    // // step controller and get actuation commands
    this->controller.controllerStep(
        torqueCommand_nm,
        brakeCommand_mps2,
        this->setSpeed, 
        egoSpeed_mps, 
        leadExists,
        leadXPos,
        leadXVel,
        leadYPos,
        leadYVel
    );

    // set actuation commands to vehicle
    this->env->setCavEnable(true);
    this->env->setCavLongitudinalCommands(torqueCommand_nm, brakeCommand_mps2);

}