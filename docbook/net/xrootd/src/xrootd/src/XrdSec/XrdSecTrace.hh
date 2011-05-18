#ifndef ___SEC_TRACE_H___
#define ___SEC_TRACE_H___
/******************************************************************************/
/*                                                                            */
/*                        X r d S e c T r a c e . h h                         */
/*                                                                            */
/* (C) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*               DE-AC03-76-SFO0515 with the Deprtment of Energy              */
/******************************************************************************/
  
//         $Id$

#include "XrdOuc/XrdOucTrace.hh"

#ifndef NODEBUG

#include "XrdSys/XrdSysHeaders.hh"

#define QTRACE(act) SecTrace->What & TRACE_ ## act

#define TRACE(act, x) \
        if (QTRACE(act)) \
           {SecTrace->Beg(epname,tident); cerr <<x; SecTrace->End();}

#define DEBUG(y) if (QTRACE(Debug)) \
                    {SecTrace->Beg(epname); cerr <<y; SecTrace->End();}
#define EPNAME(x) static const char *epname = x;

#else

#define  TRACE(x, y)
#define QTRACE(x)
#define DEBUG(x)
#define EPNAME(x)

#endif

// Trace flags
//
#define TRACE_ALL      0x000f
#define TRACE_Authenxx 0x0007
#define TRACE_Authen   0x0004
#define TRACE_Debug    0x0001

#endif
