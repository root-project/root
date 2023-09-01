
//
// These are needed to handle backward compatibility
//
#include "XrdProofdXrdVers.h"

#if ROOTXRDVERS < ROOT_OldXrdOuc
#define XrdSysPthread           XrdOucPthread
#include "XrdOuc/XrdOucPthread.hh"
#else
#include "XrdSys/XrdSysPthread.hh"
#endif


