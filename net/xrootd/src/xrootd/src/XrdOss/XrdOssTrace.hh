#ifndef _XRDOSS_TRACE_H
#define _XRDOSS_TRACE_H
/******************************************************************************/
/*                                                                            */
/*                        X r d O s s T r a c e . h h                         */
/*                                                                            */
/* (C) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC03-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/

//         $Id$

#include "XrdOuc/XrdOucTrace.hh"

// Trace flags
//
#define TRACE_ALL       0x0fff
#define TRACE_Opendir   0x0001
#define TRACE_Open      0x0002
#define TRACE_AIO       0x0004
#define TRACE_Debug     0x0800

#ifndef NODEBUG

#include "XrdSys/XrdSysHeaders.hh"

#define QTRACE(act) OssTrace.What & TRACE_ ## act

#define TRACE(act, x) \
        if (QTRACE(act)) \
           {OssTrace.Beg(epname,tident); cerr <<x; OssTrace.End();}

#define TRACEReturn(type, ecode, msg) \
               {TRACE(type, "err " <<ecode <<msg); return ecode;}

#define DEBUG(y) if (QTRACE(Debug)) \
                    {OssTrace.Beg(epname); cerr <<y; OssTrace.End();}

#define EPNAME(x) static const char *epname = x;

#else

#define DEBUG(x)
#define QTRACE(x) 0
#define TRACE(x, y)
#define TRACEReturn(type, ecode, msg) return ecode
#define EPNAME(x)

#endif
#endif
