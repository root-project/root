#ifndef _XRDFRM_TRACE_H
#define _XRDFRM_TRACE_H
/******************************************************************************/
/*                                                                            */
/*                        X r d F r m T r a c e . h h                         */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include "XrdSys/XrdSysError.hh"
#include "XrdOuc/XrdOucTrace.hh"

#define TRACE_ALL       0xffff
#define TRACE_Debug     0x0001

#ifndef NODEBUG

#include "XrdSys/XrdSysHeaders.hh"

#define QTRACE(act) Trace.What & TRACE_ ## act

#define DEBUGR(y) if (Trace.What & TRACE_Debug) \
                  {Trace.Beg(epname, Req.User); cerr <<y; Trace.End();}

#define DEBUG(y) if (Trace.What & TRACE_Debug) TRACEX(y)

#define TRACE(x,y) if (Trace.What & TRACE_ ## x) TRACEX(y)

#define TRACER(x,y) if (Trace.What & TRACE_ ## x) \
                       {Trace.Beg(epname, Req.User); cerr <<y; Trace.End();}

#define TRACEX(y) {Trace.Beg(0,epname); cerr <<y; Trace.End();}
#define EPNAME(x) static const char *epname = x;

#else

#define DEBUG(y)
#define TRACE(x, y)
#define EPNAME(x)

#endif

#define VMSG(a,...) if (Config.Verbose) Say.Emsg(a,__VA_ARGS__);

#define VSAY(a,...) if (Config.Verbose) Say.Say(a,__VA_ARGS__);

namespace XrdFrm
{
extern XrdSysError  Say;
extern XrdOucTrace  Trace;
}
#endif
