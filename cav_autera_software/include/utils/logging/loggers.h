#pragma once

#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>
#include <string>

using namespace std;

typedef pair<int, double> timeseriesEntry_t; // tuples of (time [sec], signal value)
typedef vector<timeseriesEntry_t> timeseries_t;

enum InterpolationMethod {
    HOLD_LAST,
    INTERPOLATE
};

class Loggers {

public:

    int sampleRate_Hz;
    int latestTimestep;
    string scenarioName;
    string saveLocation;
    unordered_map<string, timeseries_t> signals;
    InterpolationMethod interpolationType;

    Loggers(string scenarioName = "unknown", int sampleRate_Hz = 100, InterpolationMethod interpolationType = HOLD_LAST, string filename = "loggers.csv");

    void updateSignal(string sigName, double time_s, double value);
    void updateSignalTimestep(string sigName, int timestep, double value);

    timeseries_t* getSignalHistory(string sigName);
    double getSignalAtTimestep(string sigName, int timestep_s);
    double getSignalAtTime(string sigName, double timestep_s);

    void writeLogFileCsv();

};

// common loggers object for any cpp files to include and use (100Hz by default)
extern Loggers logger100Hz;

// helper macros
#define LOG_SIGNAL(name, time, value) logger100Hz.updateSignal(name, time, value);
#define LOG_SIGNAL_TS(name, time, value) logger100Hz.updateSignalTimestep(name, time, value);