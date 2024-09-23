#include <iostream>
#include "besee_utils.h"
#include "utils/logging/loggers.h"

/*
Entry point for besee! Invoke from command line like so:
./besee scenarioName endTime outputCsvName

Args:
scenarioName: a scenario name defined in scenarios.yaml [optional, default value oval_track_lead_vehicle_20mps]
endTime: how long to run the sim in seconds [optional, default value 100]
outputCsvName: name of logging csv to output [optional, default value loggers_<scenarioName>.csv]
*/

int main(int argc, char** argv) {

    std::cout << "Besee Time!" << std::endl;

    // if user passed in a scenario name, load that. Otherwise, load oval track with lead vehicle
    //string configScenarioname = "oval_track_lead_vehicle_20mps";
    string configScenarioname = "open_straight_road";
    if (argc >= 2) { configScenarioname = argv[1]; }

    // if user passed in an end time, set that. Otherwise, run for 100 sec
    float endTime = 100;
    if (argc >= 3) { endTime = atof(argv[2]); }

    // if user passed in a location to save loggers csv, set that. Otherwise, set location to "loggers_<scenarioName>.csv"
    string csvSaveLocation = "loggers_" + configScenarioname;
    if (argc >= 4) { csvSaveLocation = argv[3]; }
    csvSaveLocation += ".csv";

    // set signal logger parameters
    logger100Hz.scenarioName = configScenarioname;
    logger100Hz.saveLocation = PROJECT_BASE + "/loggers/" + csvSaveLocation;

    // construct besee core object
    BeseeCore besee(configScenarioname);

    // run besee!
    std::cout << "Running scenario " << configScenarioname << " for " << endTime << " seconds!" << std::endl;
    besee.runSim(endTime, 0);
    std::cout << "Saved logging info to loggers/" << csvSaveLocation << "!" << std::endl;
    
}
