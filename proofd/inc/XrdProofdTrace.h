// @(#)root/proofd:$Name:  $:$Id: XrdProofdTrace.h,v 1.3 2006/04/18 10:34:35 rdm Exp $
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
#define TRACE_DEBUG     0x0001
#define TRACE_EMSG      0x0002
#define TRACE_FS        0x0004
#define TRACE_LOGIN     0x0008
#define TRACE_MEM       0x0010
#define TRACE_REQ       0x0020
#define TRACE_REDIR     0x0040
#define TRACE_RSP       0x0080
#define TRACE_SCHED     0x0100
#define TRACE_STALL     0x0200

//#define NODEBUG
#ifndef NODEBUG

#ifndef ROOT_Riosfwd
#include "Riosfwd.h"
#endif
#include "XrdOuc/XrdOucTrace.hh"

#define PRINT(x) \
   {XrdProofdTrace->Beg(TRACEID);   cerr <<x; XrdProofdTrace->End();}

#define TRACE(act, x) \
   if (XrdProofdTrace->What & TRACE_ ## act) \
      {XrdProofdTrace->Beg(TRACEID);   cerr <<x; XrdProofdTrace->End();}

#define TRACEI(act, x) \
   if (XrdProofdTrace->What & TRACE_ ## act) \
      {XrdProofdTrace->Beg(TRACEID,TRACELINK->ID); cerr <<x; XrdProofdTrace->End();}

#define TRACEP(act, x) \
   if (XrdProofdTrace->What & TRACE_ ## act) \
      {XrdProofdTrace->Beg(TRACEID,TRACELINK->ID,RESPONSE.ID()); cerr <<x; \
       XrdProofdTrace->End();}

#define TRACES(act, x) \
   if (XrdProofdTrace->What & TRACE_ ## act) \
      {XrdProofdTrace->Beg(TRACEID,TRACELINK->ID,TRSID); cerr <<x; \
       XrdProofdTrace->End();}

#define TRACING(x) XrdProofdTrace->What & x

#else

#define TRACE(act,x)
#define TRACEI(act,x)
#define TRACEP(act,x)
#define TRACES(act,x)
#define TRACING(x) 0
#endif

#endif
