// $Id$
#ifndef ___SUT_TRACE_H___
#define ___SUT_TRACE_H___
/******************************************************************************/
/*                                                                            */
/*                        X r d S u t T r a c e . h h                         */
/*                                                                            */
/* (C) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*               DE-AC03-76-SFO0515 with the Deprtment of Energy              */
/******************************************************************************/
#ifndef ___OUC_TRACE_H___
#include "XrdOuc/XrdOucTrace.hh"
#endif
#ifndef ___SUT_AUX_H___
#include "XrdSut/XrdSutAux.hh"
#endif

#ifndef NODEBUG

#include "XrdSys/XrdSysHeaders.hh"

#define QTRACE(act) (sutTrace && (sutTrace->What & sutTRACE_ ## act))
#define PRINT(y)    {if (sutTrace) {sutTrace->Beg(epname); \
                                    cerr <<y; sutTrace->End();}}
#define TRACE(act,x) if (QTRACE(act)) PRINT(x)
#define DEBUG(y)     TRACE(Debug,y)
#define EPNAME(x)    static const char *epname = x;

#else

#define QTRACE(x)
#define  PRINT(x)
#define  TRACE(x,y)
#define  DEBUG(x)
#define EPNAME(x)

#endif

//
// For error logging and tracing
extern XrdOucTrace *sutTrace;

#endif
