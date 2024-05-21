
//
// These are needed to handle backward compatibility
//
#include "XrdProofdXrdVers.h"

#if ROOTXRDVERS < ROOT_OldXrdOuc
#define XrdSysLogger           XrdOucLogger
#include "XrdOuc/XrdOucLogger.hh"
#else
#include "XrdSys/XrdSysLogger.hh"
#endif


