#ifndef __XRDOLBSTATE_H_
#define __XRDOLBSTATE_H_
/******************************************************************************/
/*                                                                            */
/*                        X r d O l b S t a t e . h h                         */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include "XrdSys/XrdSysPthread.hh"
#include "XrdOlb/XrdOlbTypes.hh"

class XrdOlbState
{
public:

void  Calc(int how, int nosState, int susState);

void *Monitor();

void  Sync(SMask_t mmask, int nosState, int susState);

      XrdOlbState();
     ~XrdOlbState() {}

private:

XrdSysSemaphore mySemaphore;
XrdSysMutex     myMutex;

int             numSuspend;
int             numStaging;
int             curState;
int             Changes;
};

namespace XrdOlb
{
extern    XrdOlbState OlbState;
}
#endif
