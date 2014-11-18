// $Id$
#ifndef ___SECGSI_TRACE_H___
#define ___SECGSI_TRACE_H___
/******************************************************************************/
/*                                                                            */
/*                    X r d S e c g s i T r a c e . h h                       */
/*                                                                            */
/* (C) 2005  G. Ganis, CERN                                                   */
/*                                                                            */
/******************************************************************************/

#include <XrdOuc/XrdOucTrace.hh>

#ifndef NODEBUG

#include "XrdSys/XrdSysHeaders.hh"

#define QTRACE(act) (gsiTrace && (gsiTrace->What & TRACE_ ## act))
#define PRINT(y)    {if (gsiTrace) {gsiTrace->Beg(epname); \
                                       cerr <<y; gsiTrace->End();}}
#define TRACE(act,x) if (QTRACE(act)) PRINT(x)
#define NOTIFY(y)    TRACE(Debug,y)
#define DEBUG(y)     TRACE(Authen,y)
#define EPNAME(x)    static const char *epname = x;

#else

#define QTRACE(x)
#define  PRINT(x)
#define  TRACE(x,y)
#define NOTIFY(x)
#define  DEBUG(x)
#define EPNAME(x)

#endif

#define TRACE_ALL      0x000f
#define TRACE_Dump     0x0004
#define TRACE_Authen   0x0002
#define TRACE_Debug    0x0001

//
// For error logging and tracing
extern XrdOucTrace *gsiTrace;

#endif
