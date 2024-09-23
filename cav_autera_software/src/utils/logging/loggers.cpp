#include <cmath>
#include <algorithm>

#include "utils/logging/loggers.h"
#include "utils/csv/csv_utils.h"

// common loggers object
Loggers logger100Hz;

Loggers::Loggers(string scenarioName, int sampleRate_Hz, InterpolationMethod interp, string csvSaveLoc) { 
    this->scenarioName = scenarioName;
    this->sampleRate_Hz = sampleRate_Hz;
    this->interpolationType = interp;
    this->latestTimestep = 0;
    this->saveLocation = csvSaveLoc;
}

void Loggers::updateSignal(string sigName, double time_s, double value) {
    this->updateSignalTimestep(sigName, (int)(time_s * sampleRate_Hz), value);
}

void Loggers::updateSignalTimestep(string sigName, int timestep, double value) {

    // check to see if the signal missed any timesteps
    if (timestep > 0 && this->signals[sigName].size() < timestep) {

        // edge case: if vector is uninitialized, set initial default value to 0
        if (this->signals[sigName].size() == 0) {
            this->signals[sigName].push_back(timeseriesEntry_t(0, 0));
        }

        int lastRecordedTs = (--this->signals[sigName].end())->first; 
        double lastRecordedValue = (--this->signals[sigName].end())->second; 

        // iterate through all missing timesteps between last recorded + 1 and new timestep
        for (int currTs = lastRecordedTs + 1; currTs < timestep; currTs++) {
            
            switch (this->interpolationType) {
            case HOLD_LAST:
                // append a new signal entry with the last recorded value
                this->signals[sigName].push_back(timeseriesEntry_t(currTs, lastRecordedValue));
                break;

            case INTERPOLATE:
                // linearly interpolate between last recorded value and new value -> y = y0 + (x - x0)/(x1 - x0) * (y1 - y0)
                // eg. last (x0, y0) = (1, 2), new (x1, y1) = (6, 12)
                // at t2, value = 2 + ((2 - 1)/(6 - 1))*(12 - 2)
                // value = 2 + (1/5)*10 = 4
                double currValue = lastRecordedValue + 
                    ((double)(currTs - lastRecordedTs) / (double)(timestep - lastRecordedTs)) * 
                    (value - lastRecordedValue);
                this->signals[sigName].push_back(timeseriesEntry_t(currTs, currValue));
                break;
        
            }
        }
    }

    // append most recent key value pair
    this->signals[sigName].push_back(timeseriesEntry_t(timestep, value));

    // update latest timestep if necessary
    if (timestep > this->latestTimestep) { this->latestTimestep = timestep; }

}

timeseries_t* Loggers::getSignalHistory(string sigName) {
    return &this->signals[sigName];
}

double Loggers::getSignalAtTimestep(string sigName, int timestep) {
    auto sigHistory = this->getSignalHistory(sigName);
    return sigHistory->at(timestep).second;
}

double Loggers::getSignalAtTime(string sigName, double time_s) {
    return getSignalAtTimestep(sigName, (int)(time_s * sampleRate_Hz));
}

void Loggers::writeLogFileCsv() {

    // firstly, iterate through all columns and ensure that they are all the same length. if not, pad with last recorded value
    for (auto sig : this->signals) {
        if ((sig.second.end()-1)->first < this->latestTimestep) {
            double lastValue = (sig.second.end()-1)->second;
            this->updateSignalTimestep(sig.first, this->latestTimestep, lastValue);
        }
    }

    // initialize csv writer
    // fix number of rows to be equal to the number of signals + 1 for time or 2 if there are no signals logged
    CSVWriter csv(",");
    int numRows = std::max<int>(2, this->signals.size() + 1);
    csv.enableAutoNewRow(numRows);

    // first row is scenario name
    csv << "Scenario";
    csv << this->scenarioName;
    for (int i = 2; i < numRows; i++) { csv << ""; } // fill remaining rows with blank entries

    // second row is all signal names
    csv << "time";
    for (auto& sig : this->signals) {
        csv << (sig.first);
    }

    // every subsequent row is time in the first column, then the value of each signal
    for (int currTs = 0; currTs < this->latestTimestep; currTs++) {
        csv << (double)currTs / (double)this->sampleRate_Hz;
        for (auto& sig : this->signals) {
            csv << this->getSignalAtTimestep(sig.first, currTs);
        }
    }

    // write csv file
    csv.writeToFile(this->saveLocation);

}