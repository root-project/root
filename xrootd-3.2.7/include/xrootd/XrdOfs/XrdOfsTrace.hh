#ifndef ___OFS_TRACE_H___
#define ___OFS_TRACE_H___
/******************************************************************************/
/*                                                                            */
/*                        X r d O f s T r a c e . h h                         */
/*                                                                            */
/* (C) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*               DE-AC03-76-SFO0515 with the Deprtment of Energy              */
/******************************************************************************/
  
//         $Id$

#ifndef NODEBUG

#include "XrdSys/XrdSysHeaders.hh"
#include "XrdOfs/XrdOfs.hh"

#define GTRACE(act)         OfsTrace.What & TRACE_ ## act

#define TRACES(x) \
        {OfsTrace.Beg(epname,tident); cerr <<x; OfsTrace.End();}

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
#define TRACE_MOST     0x3fcd
#define TRACE_ALL      0xffff
#define TRACE_opendir  0x0001
#define TRACE_readdir  0x0002
#define TRACE_closedir TRACE_opendir
#define TRACE_delay    0x0400
#define TRACE_dir      TRACE_opendir | TRACE_readdir | TRACE_closedir
#define TRACE_open     0x0004
#define TRACE_qscan    0x0008
#define TRACE_close    TRACE_open
#define TRACE_read     0x0010
#define TRACE_redirect 0x0800
#define TRACE_write    0x0020
#define TRACE_IO       TRACE_read | TRACE_write | TRACE_aio
#define TRACE_exists   0x0040
#define TRACE_chmod    TRACE_exists
#define TRACE_getmode  TRACE_exists
#define TRACE_getsize  TRACE_exists
#define TRACE_remove   0x0080
#define TRACE_rename   TRACE_remove
#define TRACE_sync     0x0100
#define TRACE_truncate 0x0200
#define TRACE_fsctl    0x0400
#define TRACE_getstats 0x0800
#define TRACE_mkdir    0x1000
#define TRACE_stat     0x2000
#define TRACE_aio      0x4000
#define TRACE_debug    0x8000

#endif
