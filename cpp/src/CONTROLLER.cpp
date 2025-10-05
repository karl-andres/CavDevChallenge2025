#include "controller.h"
#include <iostream>
#include <cmath>
#include <algorithm>

// Constructor
CaccController::CaccController() {
    // nothing special
}

// Function called every timestep by BESEE
void CaccController::controllerStep(
        double &torqueCommand_nm,
        double &brakeCommand_mps2,
        double setSpeed,
        double egoSpeed_mps,
        bool leadExists,
        double leadXPos,
        double leadXVel,
        double leadYPos,
        double leadYVel)
{
    // --- Parameters ---
    const double dt = 0.02;                // 50 Hz
    const double kp_gap = 0.4;             // gain for gap error
    const double kp_speed = 1.5;           // gain for speed error
    const double kd = 0.05;                // derivative gain (smoothing)
    const double timeHeadway = 1.5;        // seconds
    const double minGap = 8.0;             // meters
    const double maxAccel = 2.5;           // m/s^2
    const double maxDecel = -6.0;          // m/s^2
    const double torqueToAccel = 300.0;    // Nm per m/s^2

    static double prevAccel = 0.0;

    // --- Ensure valid ego speed ---
    if (egoSpeed_mps < 0.0 || std::isnan(egoSpeed_mps)) {
        egoSpeed_mps = 0.0;
    }

    double accelerationCmd = 0.0;

    if (leadExists) {
        // --- Compute FDCW minimum following distance ---
        double leadSpeed_mph = leadXVel * 2.237;
        double fdcw_min = 2.8 * std::pow(leadSpeed_mph, 0.45) + 8.0;
        if (fdcw_min < minGap) fdcw_min = minGap;

        // --- Desired gap (spacing policy) ---
        double desiredGap = std::max(fdcw_min, egoSpeed_mps * timeHeadway + minGap);

        // --- Gap error ---
        double gapError = leadXPos - desiredGap;

        if (gapError < 0.0) {
            // ❌ Too close → brake to restore gap
            accelerationCmd = kp_gap * gapError;
        } else {
            // ✅ Safe → follow the lead’s speed
            // Too close → brake to restore gap
            accelerationCmd = kp_gap * gapError;
        } else {
            // Safe → follow the lead’s speed
            accelerationCmd = kp_speed * (leadXVel - egoSpeed_mps);
        }
    } else {
        // No lead vehicle → just track set speed
        accelerationCmd = kp_speed * (setSpeed - egoSpeed_mps);
    }

    // --- Derivative smoothing ---
    double derivative = (accelerationCmd - prevAccel) / dt;
    accelerationCmd += kd * derivative;
    prevAccel = accelerationCmd;

    // --- Clamp acceleration ---
    if (accelerationCmd > maxAccel) accelerationCmd = maxAccel;
    if (accelerationCmd < maxDecel) accelerationCmd = maxDecel;

    // --- Convert to torque/brake ---
    if (accelerationCmd >= 0) {
        torqueCommand_nm = accelerationCmd * torqueToAccel;
        brakeCommand_mps2 = 0.0;
    } else {
        torqueCommand_nm = 0.0;
        brakeCommand_mps2 = -accelerationCmd; // positive decel
    }

    // --- Safety clamps ---
    if (torqueCommand_nm > 4500.0) torqueCommand_nm = 4500.0;
    if (torqueCommand_nm < 0.0) torqueCommand_nm = 0.0;
    if (brakeCommand_mps2 > 8.0) brakeCommand_mps2 = 8.0;
    if (brakeCommand_mps2 < 0.0) brakeCommand_mps2 = 0.0;

    // Debug (optional)
    // std::cout << "Ego=" << egoSpeed_mps
    //           << " Lead=" << leadXVel
    //           << " GapError=" << (leadXPos - desiredGap)
    //           << " Accel=" << accelerationCmd
    //           << " Torque=" << torqueCommand_nm
    //           << " Brake=" << brakeCommand_mps2
    //           << std::endl;
}