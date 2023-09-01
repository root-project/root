
//
// These are needed when we build the XROOTD-related plugins with an
// external XROOTD relased before the full move XrdOuc to XrdSys
//
#ifndef ROOT_XrdSysToOuc
#define ROOT_XrdSysToOuc

#include "XrdProofdXrdVers.h"

#if ROOTXRDVERS < ROOT_OldXrdOuc

#define XrdSysError           XrdOucError
#define XrdSysLogger          XrdOucLogger

#define XrdSysMutex           XrdOucMutex
#define XrdSysMutexHelper     XrdOucMutexHelper
#define XrdSysRecMutex        XrdOucRecMutex
#define XrdSysSemWait         XrdOucSemWait
#define XrdSysThread          XrdOucThread

#define XrdSysPlugin          XrdOucPlugin

#define XrdSysTimer           XrdOucTimer

#endif
#endif


