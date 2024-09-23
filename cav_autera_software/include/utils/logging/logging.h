/***************************************************************************************************
 *  Autera Logging Header.
 **************************************************************************************************/

#pragma once

#include "utils/logging/easylogging++.h"

#define AUTERA_VERBOSE VLOG(1)
#define AUTERA_DEBUG LOG(DEBUG)
#define AUTERA_INFO LOG(INFO)
#define AUTERA_WARNING LOG(WARNING)
#define AUTERA_ERROR LOG(ERROR)
#define AUTERA_FATAL LOG(FATAL)