#pragma once

class CaccController {

public:
    // If you want to save additional variables in memory, define them here
    int x;
    // CACC controller parameters
    double desiredTimeHeadway;     // seconds
    double minFollowingDistance;  // meters
    double maxFollowingDistance;  // meters

    // PID controller gains
    double kp_speed;    // proportional gain for speed control
    double ki_speed;    // integral gain for speed control
    double kd_speed;    // derivative gain for speed control

    double kp_distance; // proportional gain for distance control
    double ki_distance; // integral gain for distance control
    double kd_distance; // derivative gain for distance control

    // Controller state
    double prevSpeedError;
    double prevDistanceError;
    double speedErrorIntegral;
    double distanceErrorIntegral;

    // Vehicle limits
    double maxTorque;  // Nm
    double maxBrake;
    
    // Constructor
    CaccController();
    // No need to edit this function signature
    void controllerStep(
        double &torqueCommand_nm,
        double &brakeCommand_mps2,
        double setSpeed, 
        double egoSpeed_mps, 
        bool leadExists,
        double leadXPos,
        double leadXVel,
        double leadYPos,
        double leadYVel
    );
};