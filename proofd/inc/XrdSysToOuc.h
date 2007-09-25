
//
// These are needed when we build the XROOTD-related plugins with an
// external XROOTD relased before the full move XrdOuc to XrdSys
//

#define XrdSysError           XrdOucError
#define XrdSysLogger          XrdOucLogger

#define XrdSysMutex           XrdOucMutex
#define XrdSysMutexHelper     XrdOucMutexHelper
#define XrdSysRecMutex        XrdOucRecMutex
#define XrdSysSemWait         XrdOucSemWait
#define XrdSysThread          XrdOucThread

#define XrdSysPlugin          XrdOucPlugin

#define XrdSysTimer           XrdOucTimer


