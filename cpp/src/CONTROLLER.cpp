#include "controller.h"
#include <iostream>

// Constructor
CaccController::CaccController() {
    // If you want to save any variables in memory (e.g. your constants for a PID controller),
    // you will need to add them to the cpp/include/controller.h file and set their values here
    x = 10;
}

// Function that is called every timestep
// It calculates the torque and break commands and passes them back to BESEE
void CaccController::controllerStep(
        // torqueCommand_nm and brakeCommand_mps2 are passed by reference so they are basically the return values of your function
        // See: https://www.geeksforgeeks.org/cpp-functions-pass-by-reference/
        double &torqueCommand_nm,
        double &brakeCommand_mps2,

        // The target speed you are aiming to reach (assuming no lead vehicle)
        double setSpeed, 
        // Your current speed
        double egoSpeed_mps, 
        // If there is a vehicle in front
        bool leadExists,
        // x, x velocity, y, y velocity of the lead vehicle (all 0 if there is no lead)
        double leadXPos,
        double leadXVel,
        double leadYPos,
        double leadYVel)
{
    /*
    YOUR CODE GOES HERE!
    */
   
    // Default outputs
    torqueCommand_nm = 5000;
    brakeCommand_mps2 = 0;
}