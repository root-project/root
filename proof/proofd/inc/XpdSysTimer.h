
//
// These are needed to handle backward compatibility
//
#include "XrdProofdXrdVers.h"

#if ROOTXRDVERS < ROOT_OldXrdOuc
#define XrdSysTimer           XrdOucTimer
#include "XrdOuc/XrdOucTimer.hh"
#else
#include "XrdSys/XrdSysTimer.hh"
#endif


