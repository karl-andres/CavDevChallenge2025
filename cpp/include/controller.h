#pragma once

class CaccController {

public:
    // If you want to save additional variables in memory, define them here
    int x;
    
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