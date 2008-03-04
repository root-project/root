#ifndef _XRDOLB_TRACE_H
#define _XRDOLB_TRACE_H
/******************************************************************************/
/*                                                                            */
/*                        X r d O l b T r a c e . h h                         */
/*                                                                            */
/* (C) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC03-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/

//         $Id$

#include "XrdSys/XrdSysError.hh"
#include "XrdOuc/XrdOucTrace.hh"

#define TRACE_ALL   0x0007
#define TRACE_Debug 0x0001
#define TRACE_Stage 0x0002
#define TRACE_Defer 0x0004

#ifndef NODEBUG

#ifndef XRDOLBTRACETYPE
#define XRDOLBTRACETYPE .
#endif

#include <iostream.h>
#include "XrdOuc/XrdOucTrace.hh"

#define DEBUG(y) if (Trace XRDOLBTRACETYPE What & TRACE_Debug) TRACEX(y)

#define TRACE(x,y) if (Trace XRDOLBTRACETYPE What & TRACE_ ## x) TRACEX(y)

#define TRACEX(y) {Trace XRDOLBTRACETYPE Beg(0,epname); cerr <<y; Trace XRDOLBTRACETYPE End();}

#define EPNAME(x) const char *epname = x;

#else

#define DEBUG(y)
#define TRACE(x, y)
#define EPNAME(x)

#endif

namespace XrdOlb
{
extern    XrdSysError Say;
extern    XrdOucTrace Trace;
}
#endif
