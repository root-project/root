// $Id$
#ifndef ___SECSSL_TRACE_H___
#define ___SECSSL_TRACE_H___
/******************************************************************************/
/*                                                                            */
/*                    X r d S e c g s i T r a c e . h h                       */
/*                                                                            */
/* (C) 2008  A.J. Peters, CERN                                                */
/*                                                                            */
/******************************************************************************/

#include <XrdOuc/XrdOucTrace.hh>

#ifndef NODEBUG

#include <iostream>

#define QTRACE(act) (SSLxTrace && (SSLxTrace->What & TRACE_ ## act))
#define PRINT(y)    {if (SSLxTrace) {SSLxTrace->Beg(epname); \
                                       cerr <<y; SSLxTrace->End();}}
#define TRACE(act,x) if (QTRACE(act)) PRINT(x)
#define DEBUG(y)     TRACE(Debug,y)
#define EPNAME(x)    const char *epname = x;

#else

#define QTRACE(x)
#define  PRINT(x)
#define  TRACE(x,y)
#define  DEBUG(x)
#define EPNAME(x)

#endif

#define TRACE_ALL      0x000f
#define TRACE_Authenxx 0x0007
#define TRACE_Authen   0x0004
#define TRACE_Debug    0x0001
#define TRACE_Identity 0x0002

#endif
