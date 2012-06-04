// @(#)root/proofd:$Id$
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

//
// Trace flags
//
// Global mask
#define TRACE_ALL       0xff7f

// Levels
#define TRACE_XERR      0x0001
#define TRACE_REQ       0x0002
#define TRACE_DBG       0x0004
#define TRACE_LOGIN     0x0008
#define TRACE_FORK      0x0010
#define TRACE_MEM       0x0020
#define TRACE_HDBG      0x0040
// Bit 0x0080 reserved for future usage

// Domains
#define TRACE_DOMAINS   0xFF00
#define TRACE_RSP       0x0100
#define TRACE_AUX       0x0200
#define TRACE_CMGR      0x0400
#define TRACE_SMGR      0x0800
#define TRACE_NMGR      0x1000
#define TRACE_PMGR      0x2000
#define TRACE_GMGR      0x4000
#define TRACE_SCHED     0x8000

#ifndef NODEBUG

#ifndef ROOT_Riosfwd
#include "Riosfwd.h"
#endif
#ifndef ROOT_DllImport
#include "DllImport.h"
#endif
#include "XrdOuc/XrdOucTrace.hh"

R__EXTERN XrdOucTrace *XrdProofdTrace;

//
// Auxilliary macros
//
#define XPDDOM(d) unsigned int xpdtracingdomain = (unsigned int)(TRACE_ ## d & TRACE_ALL);
#define XPDLOC(d,x) unsigned int xpdtracingdomain = (unsigned int)(TRACE_ ## d & TRACE_ALL); \
                    const char *xpdloc = x;

#define TRACINGALL(x) (TRACE_ALL == TRACE_ ## x)
#define TRACINGERR(x) (TRACE_XERR == TRACE_ ## x)
#define TRACINGACT(x) (XrdProofdTrace && (XrdProofdTrace->What & TRACE_ ## x))
#define TRACINGDOM    (XrdProofdTrace && (XrdProofdTrace->What & xpdtracingdomain))
#define TRACING(x) (TRACINGALL(x) || TRACINGERR(x) || (TRACINGACT(x) && TRACINGDOM))

#define TRACESET(act,on) \
        if (on) { \
           XrdProofdTrace->What |= TRACE_ ## act; \
        } else { \
           XrdProofdTrace->What &= ~(TRACE_ ## act & TRACE_ALL); \
        }

#define XPDPRT(x) \
   {XrdProofdTrace->Beg("-I");   std::cerr << xpdloc <<": "<< x; XrdProofdTrace->End();}

#define XPDERR(x) \
   {XrdProofdTrace->Beg("-E");   std::cerr << xpdloc <<": "<< x; XrdProofdTrace->End();}

#define TRACE(act, x) \
   if (TRACING(act)) { \
      if (TRACINGERR(act)) { \
         XPDERR(x); \
      } else { \
         XPDPRT(x); \
      } \
   }

#define TRACEP(p, act, x) \
   if (TRACING(act)) { \
      if (TRACINGERR(act)) { \
         if (p) {\
            XrdProofdTrace->Beg("-E", 0, p->TraceID()); std::cerr << xpdloc <<": "<< x; XrdProofdTrace->End(); \
         } else {XPDERR(x);}\
      } else { \
         if (p) {\
            XrdProofdTrace->Beg("-I", 0, p->TraceID()); std::cerr << xpdloc <<": "<< x; XrdProofdTrace->End(); \
         } else {XPDPRT(x);}\
      } \
   }

#define TRACEI(id, act, x) \
   if (TRACING(act)) { \
      if (TRACINGERR(act)) { \
         if (id) {\
            XrdProofdTrace->Beg("-E", 0, id); std::cerr << xpdloc <<": "<< x; XrdProofdTrace->End(); \
         } else { XPDERR(x); }\
      } else { \
         if (id) {\
            XrdProofdTrace->Beg("-I", 0, id); std::cerr << xpdloc <<": "<< x; XrdProofdTrace->End(); \
         } else { XPDPRT(x); }\
      } \
   }

#define TRACER(r, act, x) \
   if (TRACING(act)) { \
      if (TRACINGERR(act)) { \
         if (r) {\
            XrdProofdTrace->Beg("-E", 0, r->TraceID()); std::cerr << xpdloc <<": "<< x; XrdProofdTrace->End(); \
         } else { XPDERR(x); }\
      } else { \
         if (r) {\
            XrdProofdTrace->Beg("-I", 0, r->TraceID()); std::cerr << xpdloc <<": "<< x; XrdProofdTrace->End(); \
         } else { XPDPRT(x); }\
      } \
   }

#else

// Dummy versions

#define TRACING(x) 0
#define TRACINGERR(x) (0)
#define TRACESET(act,on)
#define XPDLOC(x)
#define XPDPRT(x)
#define XPDERR(x)
#define TRACE(act, x)
#define TRACEP(p, act, x)
#define TRACEI(id, act, x)
#define TRACER(r, act, x)

#endif

#endif
