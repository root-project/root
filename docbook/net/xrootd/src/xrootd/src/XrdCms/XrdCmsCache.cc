/******************************************************************************/
/*                                                                            */
/*                        X r d C m s C a c h e . c c                         */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

// Original Version: 1.13 2007/07/12 21:57:38 abh

const char *XrdCmsCacheCVSID = "$Id$";
  
#include <stdio.h>
#include <sys/types.h>

#include "XrdCms/XrdCmsCache.hh"
#include "XrdCms/XrdCmsRRQ.hh"
#include "XrdCms/XrdCmsTrace.hh"

#include "XrdSys/XrdSysTimer.hh"

#include "Xrd/XrdJob.hh"
#include "Xrd/XrdScheduler.hh"

namespace XrdCms
{
extern XrdScheduler *Sched;
}

using namespace XrdCms;

/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
XrdCmsCache XrdCms::Cache;
  
/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/
  
class XrdCmsCacheJob : XrdJob
{
public:

void   DoIt() {Cache.Recycle(myList); delete this;}

       XrdCmsCacheJob(XrdCmsKeyItem *List)
                     : XrdJob("cache scrubber"), myList(List) {}
      ~XrdCmsCacheJob() {}

private:

XrdCmsKeyItem *myList;
};

/******************************************************************************/
/*            E x t e r n a l   T h r e a d   I n t e r f a c e s             */
/******************************************************************************/
  
void *XrdCmsStartTickTock(void *carg)
     {XrdCmsCache *myCache = (XrdCmsCache *)carg;
      return myCache->TickTock();
     }

/******************************************************************************/
/*     P u b l i c   C a c h e   M a n i p u l a t i o n   M e t h o d s      */
/******************************************************************************/
/******************************************************************************/
/* Public                        A d d F i l e                                */
/******************************************************************************/

// This method insert or updates information about a path in the cache.

// Key was found:  Location information is updated depending on mask
// mask == 0       Indicates that the information is being refreshed.
//                 Location information is nullified. The update deadline is set
//                 DLTime seconds in the future. The entry window is set to the
//                 current window to be held for a full fxhold period.
// mask != 0       Indicates that some location information is now known.
//                 Location information is updated according to the mask.
//                 For a r/w location, the deadline is satisfied and all
//                 callbacks are dispatched. For an r/o location the deadline
//                 is satisfied if no r/w callback is pending. Any r/o
//                 callback is dispatched. The Info object is ignored.

// Key not found:  A selective addition occurs, depending on Sel.Opts
// Opts !Advisory: The entry is added to the cache with location information
//                 set as passed (usually 0). The update deadline is us set to
//                 DLTtime seconds in the future. The entry window is set 
//                 to the current window.
// Opts  Advisory: The call is ignored since we do not keep information about
//                 paths that were never asked for.

// Returns True    If this is the first time location information was added
//                 to the entry.
// Returns False   Otherwise.
  
int XrdCmsCache::AddFile(XrdCmsSelect &Sel, SMask_t mask)
{
   XrdCmsKeyItem *iP;
   SMask_t xmask;
   int isrw = (Sel.Opts & XrdCmsSelect::Write), isnew = 0;

// Serialize processing
//
   myMutex.Lock();

// Check for fast path processing
//
   if (  !(iP = Sel.Path.TODRef) || !(iP->Key.Equiv(Sel.Path)))
      if ((iP = Sel.Path.TODRef = CTable.Find(Sel.Path)))
         Sel.Path.Ref = iP->Key.Ref;

// Add/Modify the entry
//
   if (iP)
      {if (!mask)
          {iP->Loc.deadline = DLTime + time(0);
           iP->Loc.hfvec = 0; iP->Loc.pfvec = 0; iP->Loc.qfvec = 0;
           iP->Loc.TOD_B = BClock;
           iP->Key.TOD = Tock;
          } else {
           xmask = iP->Loc.pfvec;
           if (Sel.Opts & XrdCmsSelect::Pending) iP->Loc.pfvec |= mask;
              else iP->Loc.pfvec &= ~mask;
           isnew = (iP->Loc.hfvec == 0) || (iP->Loc.pfvec != xmask);
           iP->Loc.hfvec |=  mask;
           iP->Loc.qfvec &= ~mask;
           if (isrw) {iP->Loc.deadline = 0;
                      if (iP->Loc.roPend || iP->Loc.rwPend)
                         Dispatch(iP, iP->Loc.roPend, iP->Loc.rwPend);
                     }
              else   {if (!iP->Loc.rwPend) iP->Loc.deadline = 0;
                      if (iP->Loc.roPend) Dispatch(iP, iP->Loc.roPend, 0);
                     }
          }
      } else if (!(Sel.Opts & XrdCmsSelect::Advisory))
                {Sel.Path.TOD = Tock;
                 if ((iP = CTable.Add(Sel.Path)))
                    {iP->Loc.pfvec    = (Sel.Opts&XrdCmsSelect::Pending?mask:0);
                     iP->Loc.hfvec    = mask;
                     iP->Loc.TOD_B    = BClock;
                     iP->Loc.qfvec    = 0;
                     iP->Loc.deadline = DLTime + time(0);
                     Sel.Path.Ref     = iP->Key.Ref;
                     Sel.Path.TODRef  = iP; isnew = 1;
                    }
                }

// All done
//
   myMutex.UnLock();
   return isnew;
}
  
/******************************************************************************/
/* Public                        D e l F i l e                                */
/******************************************************************************/

// This method removes location information from existing valid entries. If an
// existing valid entry is found, based on Sel.Opts the following occurs:

// Opts  Advisory only locate information is removed.
// Opts !Advisory if the entry has no location information it is removed from
//                the cache, if possible.

// TRUE is returned if the entry was valid but location information was cleared.
// Otherwise, FALSE is returned.
  
int XrdCmsCache::DelFile(XrdCmsSelect &Sel, SMask_t mask)
{
   XrdCmsKeyItem *iP;
   int gone4good;

// Lock the hash table
//
   myMutex.Lock();

// Look up the entry and remove server
//
   if ((iP = CTable.Find(Sel.Path)))
      {iP->Loc.hfvec &= ~mask;
       iP->Loc.pfvec &= ~mask;
       if ((gone4good = (iP->Loc.hfvec == 0))
       && (!(Sel.Opts & XrdCmsSelect::Advisory))
       && (XrdCmsKeyItem::Unload(iP) && !CTable.Recycle(iP)))
          Say.Emsg("DelFile", "Delete failed for", iP->Key.Val);
      } else gone4good = 0;

// All done
//
   myMutex.UnLock();
   return gone4good;
}
  
/******************************************************************************/
/* Public                        G e t F i l e                                */
/******************************************************************************/

// This method looks up entries in the cache. An "entry not found" condition
// holds is the entry was found but is marked as deleted.

// Entry was found: Location information is passed bask. If the update deadline
//                  has passed, it is nullified and 1 is returned. Otherwise,
//                  -1 is returned indicating a query is in progress.

// Entry not found: FALSE is returned.
  
int  XrdCmsCache::GetFile(XrdCmsSelect &Sel, SMask_t mask)
{
   XrdCmsKeyItem *iP;
   SMask_t bVec;
   int retc;

// Lock the hash table
//
   myMutex.Lock();

// Look up the entry and return location information
//
   if ((iP = CTable.Find(Sel.Path)))
      {if ((bVec = (iP->Loc.TOD_B < BClock 
                 ? getBVec(iP->Key.TOD, iP->Loc.TOD_B) & mask : 0)))
          {iP->Loc.hfvec &= ~bVec; 
           iP->Loc.pfvec &= ~bVec;
           iP->Loc.qfvec &= ~mask;
           iP->Loc.deadline = DLTime + time(0); 
           retc = -1;
          } else if (iP->Loc.deadline)
                    if (iP->Loc.deadline > time(0)) retc = -1;
                       else {iP->Loc.deadline = 0;  retc =  1;}
                    else retc = 1;
       Sel.Vec.hf      = okVec & iP->Loc.hfvec;
       Sel.Vec.pf      = okVec & iP->Loc.pfvec;
       Sel.Vec.bf      = okVec & (bVec | iP->Loc.qfvec); iP->Loc.qfvec = 0;
       Sel.Path.Ref    = iP->Key.Ref;
      } else retc = 0;

// All done
//
   myMutex.UnLock();
   Sel.Path.TODRef = iP;
   return retc;
}

/******************************************************************************/
/* Public                        U n k F i l e                                */
/******************************************************************************/
  
int XrdCmsCache::UnkFile(XrdCmsSelect &Sel, SMask_t mask)
{
   EPNAME("UnkFile");
   XrdCmsKeyItem *iP;

// Make sure we have the proper information. If so, lock the hash table
//
   myMutex.Lock();

// Look up the entry and if valid update the unqueried vector. Note that
// this method may only be called after GetFile() or AddFile() for a new entry
//
   if ((iP = Sel.Path.TODRef))
      {if (iP->Key.Equiv(Sel.Path)) iP->Loc.qfvec = mask;
          else iP = 0;
      }

// Return result
//
   myMutex.UnLock();
   DEBUG("rc=" <<(iP ? 1 : 0) <<" path=" <<Sel.Path.Val);
   return (iP ? 1 : 0);
}
  
/******************************************************************************/
/* Public                        W T 4 F i l e                                */
/******************************************************************************/
  
int XrdCmsCache::WT4File(XrdCmsSelect &Sel, SMask_t mask)
{
   EPNAME("WT4File");
   XrdCmsKeyItem *iP;
   time_t  Now;
   int     retc;

// Make sure we have the proper information. If so, lock the hash table
//
   if (!Sel.InfoP) return DLTime;
   myMutex.Lock();

// Look up the entry and if valid add it to the callback queue. Note that
// this method may only be called after GetFile() or AddFile() for a new entry
//
   if (!(iP = Sel.Path.TODRef) || !(iP->Key.Equiv(Sel.Path))) retc = DLTime;
      else if (iP->Loc.hfvec != mask)                         retc = 1;
              else {Now = time(0);                            retc = 0;
                    if (iP->Loc.deadline && iP->Loc.deadline <= Now)
                        iP->Loc.deadline = DLTime + Now;
                    Add2Q(Sel.InfoP, iP, Sel.Opts & XrdCmsSelect::Write);
                   }

// Return result
//
   myMutex.UnLock();
   DEBUG("rc=" <<retc <<" path=" <<Sel.Path.Val);
   return retc;
}
  
/******************************************************************************/
/*         P u b l i c   A d m i n i s t r a t i v e   C l a s s e s          */
/******************************************************************************/
/******************************************************************************/
/* public                         B o u n c e                                 */
/******************************************************************************/

void XrdCmsCache::Bounce(SMask_t smask, int SNum)
{

// Simply indicate that this server bounced
//
   myMutex.Lock();
   Bounced[SNum] = ++BClock;
   okVec |= smask;
   if (SNum > vecHi) vecHi = SNum;
   myMutex.UnLock();
}

/******************************************************************************/
/* Public                           D r o p                                   */
/******************************************************************************/
  
void XrdCmsCache::Drop(SMask_t smask, int SNum, int xHi)
{
   SMask_t nmask(~smask);

// Remove the node from the path list
//
   Paths.Remove(smask);

// Remove the node from the list of valid nodes
//
   myMutex.Lock();
   Bounced[SNum] = 0;
   okVec &= nmask;
   vecHi = xHi;
   myMutex.UnLock();
}

/******************************************************************************/
/* public                           I n i t                                   */
/******************************************************************************/
  
int XrdCmsCache::Init(int fxHold, int fxDelay)
{
   XrdCmsKeyItem *iP;
   pthread_t tid;

// Initialize the delay time and the bounce clock tick window
//
   DLTime = fxDelay;
   if (!(Tick = fxHold/XrdCmsKeyItem::TickRate)) Tick = 1;

// Start the clock thread
//
   if (XrdSysThread::Run(&tid, XrdCmsStartTickTock, (void *)this,
                            0, "Cache Clock"))
      {Say.Emsg("Init", errno, "start cache clock");
       return 0;
      }

// Get the first reserve of cache items
//
   iP = XrdCmsKeyItem::Alloc(0);
   XrdCmsKeyItem::Unload((unsigned int)0);
   iP->Recycle();

// All done
//
   return 1;
}

/******************************************************************************/
/* public                       T i c k T o c k                               */
/******************************************************************************/

void *XrdCmsCache::TickTock()
{
   XrdCmsKeyItem *iP;

// Simply adjust the clock and trim old entries
//
   do {XrdSysTimer::Snooze(Tick);
       myMutex.Lock();
       Tock = (Tock+1) & XrdCmsKeyItem::TickMask;
       Bhistory[Tock].Start = Bhistory[Tock].End = 0;
       iP = XrdCmsKeyItem::Unload(Tock);
       myMutex.UnLock();
       if (iP) Sched->Schedule((XrdJob *)new XrdCmsCacheJob(iP));
      } while(1);

// Keep compiler happy
//
   return (void *)0;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                                 A d d 2 Q                                  */
/******************************************************************************/
  
void XrdCmsCache::Add2Q(XrdCmsRRQInfo *Info, XrdCmsKeyItem *iP, int isrw)
{
   short Slot = (isrw ? iP->Loc.rwPend : iP->Loc.roPend);

// Add the request to the appropriate pending queue
//
   Info->Key = iP;
   Info->isRW= isrw;
   if (!(Slot = RRQ.Add(Slot, Info))) Info->Key = 0;
      else if (isrw) iP->Loc.rwPend = Slot;
               else  iP->Loc.roPend = Slot;
}

/******************************************************************************/
/*                              D i s p a t c h                               */
/******************************************************************************/
  
void XrdCmsCache::Dispatch(XrdCmsKeyItem *iP, short roQ, short rwQ)
{

// Dispach the waiting elements
//
   if (roQ) {RRQ.Ready(roQ, iP, iP->Loc.hfvec, iP->Loc.pfvec);
             iP->Loc.roPend = 0;
            }
   if (rwQ) {RRQ.Ready(rwQ, iP, iP->Loc.hfvec, iP->Loc.pfvec);
             iP->Loc.rwPend = 0;
            }
}

/******************************************************************************/
/*                               g e t B V e c                                */
/******************************************************************************/
  
SMask_t XrdCmsCache::getBVec(unsigned int TODa, unsigned int &TODb)
{
   EPNAME("getBVec");
   SMask_t BVec(0);
   long long i;

// See if we can use a previously calculated bVec
//
   if (Bhistory[TODa].End == BClock && Bhistory[TODa].Start <= TODb)
      {Bhits++; TODb = BClock; return Bhistory[TODa].Vec;}

// Calculate the new vector
//
   for (i = 0; i <= vecHi; i++)
       if (TODb < Bounced[i]) BVec |= 1ULL << i;

   Bhistory[TODa].Vec   = BVec;
   Bhistory[TODa].Start = TODb;
   Bhistory[TODa].End   = BClock;
   TODb                 = BClock;
   Bmiss++;
   if (!(Bmiss & 0xff)) DEBUG("hits=" <<Bhits <<" miss=" <<Bmiss);
   return BVec;
}

/******************************************************************************/
/*                               R e c y c l e                                */
/******************************************************************************/
  
void XrdCmsCache::Recycle(XrdCmsKeyItem *theList)
{
   XrdCmsKeyItem *iP;
   char msgBuff[100];
   int numNull, numHave, numFree, numRecycled = 0;

// Recycle the list of cache items, as needed
//
   while((iP = theList))
        {theList = iP->Key.TODRef;
         if (iP->Loc.roPend) RRQ.Del(iP->Loc.roPend, iP);
         if (iP->Loc.rwPend) RRQ.Del(iP->Loc.rwPend, iP);
         myMutex.Lock(); CTable.Recycle(iP); myMutex.UnLock();
         numRecycled++;
        }

// See if we have enough items in reserve
//
   myMutex.Lock();
   XrdCmsKeyItem::Stats(numHave, numFree, numNull);
   if (numFree < XrdCmsKeyItem::minFree)
      {myMutex.UnLock();
       if (!(numNull /= 4)) numNull = 1;
       numHave += XrdCmsKeyItem::minAlloc * numNull;
       while(numNull--)
            {myMutex.Lock();
             numFree = XrdCmsKeyItem::Replenish();
             myMutex.UnLock();
            }
      } else myMutex.UnLock();

// Log the stats
//
   sprintf(msgBuff, "%d cache items; %d allocated %d free",
           numRecycled, numHave, numFree);
   Say.Emsg("Recycle", msgBuff);
}
