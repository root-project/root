#ifndef __SYS_XSLOCK_HH__
#define __SYS_XSLOCK_HH__
/******************************************************************************/
/*                                                                            */
/*                       X r d S y s X S L o c k . h h                        */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include <errno.h>
#include "XrdSys/XrdSysPthread.hh"

// These are valid usage options
//
enum XrdSysXS_Type {xs_None = 0, xs_Shared = 1, xs_Exclusive = 2};

// This class implements the shared lock. Any number of readers are allowed
// by requesting a shared lock. Only one exclusive writer is allowed by
// requesting an exclusive lock. Up/downgrading is not supported.
//
class XrdSysXSLock
{
public:

void        Lock(const XrdSysXS_Type usage);

void      UnLock(const XrdSysXS_Type usage=xs_None);

          XrdSysXSLock()
               {cur_usage = xs_None; cur_count = 0;
                exc_wait = 0; shr_wait = 0; toggle = 0;}

         ~XrdSysXSLock();

private:

XrdSysXS_Type cur_usage;
int           cur_count;
int           exc_wait;
int           shr_wait;
int           toggle;

XrdSysMutex       LockContext;
XrdSysSemaphore   WantShr;
XrdSysSemaphore   WantExc;
};
#endif
