
//
// These are needed to handle backward compatibility
//
#include "XrdProofdXrdVers.h"

#if ROOTXRDVERS < ROOT_OldXrdOuc
#define XrdSysSemWait           XrdOucSemWait
#include "XrdOuc/XrdOucSemWait.hh"
#else
#include "XrdSys/XrdSysSemWait.hh"
#endif


