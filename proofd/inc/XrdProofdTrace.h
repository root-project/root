// @(#)root/proofd:$Name:  $:$Id: XrdProofdTrace.h,v 1.9 2007/03/20 16:16:04 rdm Exp $
// Author: G. Ganis  June 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofdTrace
#define ROOT_XrdProofdTrace

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdTrace                                                       //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
// Trace utils for xproofd.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

// Trace flags
#define TRACE_ALL       0x0fff
#define TRACE_REQ       0x0001
#define TRACE_LOGIN     0x0002
#define TRACE_ACT       0x0004
#define TRACE_RSP       0x0008
#define TRACE_MEM       0x0010
#define TRACE_DBG       0x0020
#define TRACE_XERR      0x0040
#define TRACE_FORK      0x0080
#define TRACE_HDBG      0x0100

#ifndef NODEBUG

#ifndef ROOT_Riosfwd
#include "Riosfwd.h"
#endif
#ifndef ROOT_DllImport
#include "DllImport.h"
#endif
#include "XrdOuc/XrdOucTrace.hh"

R__EXTERN XrdOucTrace *XrdProofdTrace;

// Auxilliary macro
#define TRACING(x) (XrdProofdTrace->What & TRACE_ ## x)
#define TRACESET(act,on) \
        if (on) { \
           XrdProofdTrace->What |= TRACE_ ## act; \
        } else { \
           XrdProofdTrace->What &= ~(TRACE_ ## act & TRACE_ALL); \
        }

//
// "Full-tracing" macros (pid, time, ...)
//
#define XPDPRT(x) \
   {XrdProofdTrace->Beg(TRACEID);   cerr <<x; XrdProofdTrace->End();}

#define XPDERR(x) \
   {XrdProofdTrace->Beg(TRACEID);   cerr << ">>> ERROR: "<<x; XrdProofdTrace->End();}

#define TRACE(act, x) if (TRACING(act)) XPDPRT(x)

#define TRACEI(act, x) \
   if (TRACING(act)) \
      {XrdProofdTrace->Beg(TRACEID,TRACELINK->ID); cerr <<x; XrdProofdTrace->End();}

#define TRACEP(act, x) \
   if (TRACING(act)) \
      {XrdProofdTrace->Beg(TRACEID,TRACELINK->ID,RESPONSE.ID()); cerr <<x; \
       XrdProofdTrace->End();}

#define TRACES(act, x) \
   if (TRACING(act)) \
      {XrdProofdTrace->Beg(TRACEID,TRACELINK->ID,TRSID); cerr <<x; \
       XrdProofdTrace->End();}

#define TRACESTR(act, x) \
   if (TRACING(act)) \
      {XrdProofdTrace->Beg(TRACEID,TRACELINK->ID,RESPONSE.STRID()); cerr <<x; \
       XrdProofdTrace->End();}

//
// "Minimal-tracing" macros (no pid, time, ... but avoid mutex locking)
//
#define MPRINT(h,x) {cerr << h << ": " << x << endl;}
#define MERROR(h,x) {cerr << ">>> ERROR: " << h << ": " << x << endl;}
#define MTRACE(act, h, x) if (TRACING(act)) MPRINT(h, x)

#else

// Dummy versions

#define TRACING(x) 0

#define XPDPRT(x)
#define XPDERR(x) \
#define TRACE(act,x)
#define TRACEI(act,x)
#define TRACEP(act,x)
#define TRACES(act,x)

#define MPRINT(h,x)
#define MERROR(h,x)
#define MTRACE(act,h,x)

#endif

#endif
