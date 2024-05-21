
//
// These are needed to handle backward compatibility
//
#include "XrdProofdXrdVers.h"

#if ROOTXRDVERS < ROOT_OldXrdOuc
#define XrdSysError           XrdOucError
#include "XrdOuc/XrdOucError.hh"
#else
#include "XrdSys/XrdSysError.hh"
#endif


