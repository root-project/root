
//
// These are needed to handle backward compatibility
//
#include "XrdProofdXrdVers.h"

#if ROOTXRDVERS < ROOT_OldXrdOuc
#define XrdSysPlugin           XrdOucPlugin
#include "XrdOuc/XrdOucPlugin.hh"
#else
#include "XrdSys/XrdSysPlugin.hh"
#endif


