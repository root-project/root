#ifndef ___BWM_TRACE_H___
#define ___BWM_TRACE_H___
/******************************************************************************/
/*                                                                            */
/*                        X r d B w m T r a c e . h h                         */
/*                                                                            */
/* (C) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*               DE-AC02-76-SFO0515 with the Deprtment of Energy              */
/******************************************************************************/
  
//         $Id$

#ifndef NODEBUG

#include "XrdSys/XrdSysHeaders.hh"
#include "XrdOuc/XrdOucTrace.hh"

extern XrdOucTrace BwmTrace;

#define GTRACE(act)         BwmTrace.What & TRACE_ ## act

#define TRACES(x) \
        {BwmTrace.Beg(epname,tident); cerr <<x; BwmTrace.End();}

#define FTRACE(act, x) \
   if (GTRACE(act)) \
      TRACES(x <<" fn=" << (oh->Name()))

#define XTRACE(act, target, x) \
   if (GTRACE(act)) TRACES(x <<" fn=" <<target)

#define ZTRACE(act, x) if (GTRACE(act)) TRACES(x)

#define DEBUG(x) if (GTRACE(debug)) TRACES(x)

#define EPNAME(x) static const char *epname = x;

#else

#define FTRACE(x, y)
#define GTRACE(x)    0
#define TRACES(x)
#define XTRACE(x, y, a1)
#define YTRACE(x, y, a1, a2, a3, a4, a5)
#define ZTRACE(x, y)
#define DEBUG(x)
#define EPNAME(x)

#endif

// Trace flags
//
#define TRACE_ALL      0xffff
#define TRACE_calls    0x0001
#define TRACE_delay    0x0002
#define TRACE_sched    0x0004
#define TRACE_tokens   0x0008
#define TRACE_debug    0x8000

#endif
