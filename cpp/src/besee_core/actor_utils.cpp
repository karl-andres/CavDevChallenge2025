#include <cstddef>
#include <memory>
#include <vector>

#include "utils/csv/csv_utils.h"
#include "matrix_utils.h"
#include "besee_utils.h"

Actor::Actor() {

    this->name = "undefined";
    this->length_m = 4;
    this->width_m = 2;
    this->state = Matrix13d({{0,0,0}});

}

Actor::Actor(string name, double length_m, double width_m, Matrix13d init_state) {

    this->name = name;
    this->length_m = length_m;
    this->width_m = width_m;
    this->state = init_state;

}

double Actor::getX() { return this->state(0, 0); }
double Actor::getY() { return this->state(0, 1); }
double Actor::getHeading() { return this->state(0, 2); }

DriveCycle::DriveCycle() {}

DriveCycle::DriveCycle(string filename) {
    CSVReader reader(filename);
    //this->times = VectorXd::Zero(reader.csvRows.size() - 1);
    //this->speeds_mps = VectorXd::Zero(reader.csvRows.size() - 1);

    int count = 0;
    for (auto row : reader.csvRows) {
        if (count == 0) {
            count++;
            continue;
        }
        this->times.push_back(atof(row[0].c_str()));
        this->speeds_mps.push_back(atof(row[1].c_str()));
    }

}

DriveCycle::DriveCycle(vector<double>& times, vector<double>& speeds) {
    this->times = times;
    this->speeds_mps = speeds;
}

double DriveCycle::getSpeed(double time) {
    Eigen::VectorXd times = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(this->times.data(), this->times.size());
    Eigen::VectorXd speeds = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(this->speeds_mps.data(), this->speeds_mps.size());
    return interpolate1d(times, speeds, time);
}

PathFollowingActor::PathFollowingActor(string name, Path2D path, double speed_mps, double initDist_m, double length_m, double width_m, shared_ptr<DriveCycle> driveCycle) : 
    path(path),
    speed_mps(speed_mps),
    distAlongPath(initDist_m)
{
    this->name = name;
    this->length_m = length_m;
    this->width_m = width_m;
    this->state = this->path.distanceToWaypoint(this->distAlongPath);
    this->driveCycle = driveCycle;
    this->time = 0;
}

void PathFollowingActor::update(double dt) {
    this->time += dt;
    if (this->driveCycle != nullptr) { this->speed_mps = driveCycle->getSpeed(this->time); }
    this->distAlongPath += (this->speed_mps * dt);
    this->state = this->path.distanceToWaypoint(this->distAlongPath);
}

EgoVehicle::EgoVehicle(string name, double init_x, double init_y, double init_heading, double length_m, double width_m) {
    this->name = name;
    this->length_m = length_m;
    this->width_m = width_m;
    this->state = Matrix13d({{init_x, init_y, init_heading}});
    this->init_state = this->state;
    this->reset();
}

void EgoVehicle::update(double dt) {

    // define sample rates for plant model tasks
    int BASE_SAMPLE_RATE_HZ = 2000;
    int TASK_2_SAMPLE_RATE_HZ = 100;
    int TASK_3_SAMPLE_RATE_HZ = 10;
    double TASK_4_SAMPLE_RATE_HZ = 0.2;

    // create inputs struct to feed into simulink model
    ExtU_CavLyriqModel_T inputs;
		
    inputs.Driver.SteerCmd = this->steeringWheelAngle_rad;
    inputs.Driver.AccelCmd = this->accelPedalPos_pct;
    inputs.Driver.DecelCmd = this->brakePedalPos_pct;
    inputs.Driver.GearCmd = this->gearCmd;
    inputs.Driver.IgnSwtch = false;

    inputs.CavEnable = this->cavEnable;

    inputs.CavCommands.CavAxlTrqReq_Nm = this->cavAxlTrqReq_Nm;
    inputs.CavCommands.CavBrkReq_mps2 = this->cavBrakeTrqReq_mps2;
    inputs.CavCommands.CavSteerTrq_Nm = this->cavSteeringTrqReq_Nm;
    
    // set model inputs
    this->model.setExternalInputs(&inputs);

    // determine number of timesteps to run
    int numTimesteps = (int)(dt * BASE_SAMPLE_RATE_HZ);
    int endTimestep = this->timestep + numTimesteps;

    // iterate through model for as many timesteps as required
    while (this->timestep <= endTimestep) {
        model.step0();
        if (this->timestep % (int)(BASE_SAMPLE_RATE_HZ / TASK_2_SAMPLE_RATE_HZ) == 0) { this->model.step2(); }
        if (this->timestep % (int)(BASE_SAMPLE_RATE_HZ / TASK_3_SAMPLE_RATE_HZ) == 0) { this->model.step3(); }
        if (this->timestep % (int)(BASE_SAMPLE_RATE_HZ / TASK_4_SAMPLE_RATE_HZ) == 0) { this->model.step4(); }
        this->timestep++;
    }

    // get outputs from plant
    ExtY_CavLyriqModel_T outputs = model.getExternalOutputs();
    double xPos = outputs.VehFdbk.Body.InertFrm.Cg.Disp.X;
    double yPos = outputs.VehFdbk.Body.InertFrm.Cg.Disp.Y;
    double heading = outputs.VehFdbk.Body.InertFrm.Cg.Ang.psi;

    // update position of ego vehicle in world
    this->state = this->init_state + Matrix13d({{xPos, yPos, heading}});

}

void EgoVehicle::reset() {
    this->timestep = 0;
    this->model.terminate();
    this->model.initialize();
}