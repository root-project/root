/******************************************************************************/
/*                                                                            */
/*                        X r d O l b S t a t e . c c                         */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include "XrdOlb/XrdOlbManager.hh"
#include "XrdOlb/XrdOlbState.hh"

using namespace XrdOlb;
 
/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
#define OLB_ALL_SUSPEND 1
#define OLB_ALL_NOSTAGE 2

XrdOlbState XrdOlb::OlbState;

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdOlbState::XrdOlbState() : mySemaphore(0)
{
   numSuspend = 0;
   numStaging = 0;
   curState   = OLB_ALL_NOSTAGE | OLB_ALL_SUSPEND;
   Changes    = 0;
}
 
/******************************************************************************/
/*                                  C a l c                                   */
/******************************************************************************/
  
// Warning: STMutex must be locked!

void XrdOlbState::Calc(int how, int nostg, int susp)
{
  int newState, newChanges;

// Calculate new state (depends on overlapping mutex locks)
//
   myMutex.Lock();
   if (!nostg) numStaging += how;
   if ( susp)  numSuspend += how;
   newState = (numSuspend == Manager.ServCnt ? OLB_ALL_SUSPEND : 0) |
              (numStaging ? 0 : OLB_ALL_NOSTAGE);

// If any changes are noted then we must notify all our managers
//
   if ((newChanges = (newState ^ curState)))
      {curState = newState;
       Changes |= newChanges;
       mySemaphore.Post();
      }

// All done
//
   myMutex.UnLock();
}
 
/******************************************************************************/
/*                               M o n i t o r                                */
/******************************************************************************/
  
void *XrdOlbState::Monitor()
{

// Do this forever
//
   do {mySemaphore.Wait();
       myMutex.Lock();
       if (Changes & OLB_ALL_SUSPEND) 
          Manager.Inform((curState & OLB_ALL_SUSPEND ? "suspend\n":"resume\n"));
       if (Changes & OLB_ALL_NOSTAGE) 
          Manager.Inform((curState & OLB_ALL_NOSTAGE ? "nostage\n":"stage\n"));
       Changes = 0;
       myMutex.UnLock();
      } while(1);

// All done
//
   return (void *)0;
}
  
/******************************************************************************/
/*                                  S y n c                                   */
/******************************************************************************/
  
void XrdOlbState::Sync(SMask_t mmask, int oldnos, int oldsus)
{
   int oldState, oldChanges;

// Compute the old state
//
   oldState = (oldnos ? OLB_ALL_NOSTAGE : 0);
   if (oldsus) oldState |= OLB_ALL_SUSPEND;

// If the current state does not correspond to the incomming state, notify
// the mansger of the actual new state.
//
   myMutex.Lock();
   if ((oldChanges = oldState ^ curState))
      {if (oldChanges & OLB_ALL_SUSPEND) Manager.Inform(mmask,
          (curState & OLB_ALL_SUSPEND ? "suspend\n" : "resume\n"));
       if (oldChanges & OLB_ALL_NOSTAGE) Manager.Inform(mmask,
          (curState & OLB_ALL_NOSTAGE ? "nostage\n" : "stage\n"));
      }
   myMutex.UnLock();
}
