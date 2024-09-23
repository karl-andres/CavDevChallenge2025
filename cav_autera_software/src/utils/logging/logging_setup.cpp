/***************************************************************************************************
 *  Autera Logging Set-up Source.
 **************************************************************************************************/

#include "utils/logging/logging_setup.h"
#include "utils/logging/logging.h"

INITIALIZE_EASYLOGGINGPP

void configureLogging()
{
    // Set the default logger configuration from the logging calibration file
    el::Configurations conf("../calibrations/logging.conf");
    el::Loggers::reconfigureAllLoggers(conf);
}