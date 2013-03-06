// @(#)root/proofd:$Id$
// Author: Gerardo Ganis  Feb 2013

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XpdObject                                                            //
//                                                                      //
// Authors: G. Ganis, CERN, 2013                                        //
//                                                                      //
// Auxilliary class to stack protocols.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XpdObject.h"
#include "XrdProofdAux.h"
#include "XrdProofdProtocol.h"
#include "Xrd/XrdScheduler.hh"
#include "XrdOuc/XrdOucTrace.hh"

//_______________________________________________________________________
XrdProofdProtocol *XpdObjectQ::Pop()
{
   // Pop up a protocol object 

   XpdObject *node;
   fQMutex.Lock();
   if ((node = fFirst)) {
      fFirst = fFirst->fNext;
      fCount--;
   }
   fQMutex.UnLock();
   if (node) return node->fItem;
   return (XrdProofdProtocol *)0;
}

//_______________________________________________________________________
void XpdObjectQ::Push(XpdObject *node)
{
   // Push back a protocol

   node->fQTime = fCurage;
   fQMutex.Lock();
   if (fCount >= fMaxinQ) {
      delete node->fItem;
   } else {
      node->fNext = fFirst;
      fFirst = node;
      fCount++;
   }
   fQMutex.UnLock();
}

//_______________________________________________________________________
void XpdObjectQ::Set(int inQMax, time_t agemax)
{
   // Lock the data area and set the values

   fQMutex.Lock();
   fMaxinQ = inQMax; fMaxage = agemax;
   if (!(fMininQ = inQMax/2)) fMininQ = 1;
   fQMutex.UnLock();

   // Schedule ourselves using the new values
   if (agemax > 0)
      fSched->Schedule((XrdJob *)this, agemax + time(0));
}

//_______________________________________________________________________
void XpdObjectQ::DoIt()
{
   // Process method

   XpdObject *pp, *p;
   int oldcnt, agemax;

   // Lock the anchor and see if we met the threshold for deletion
   //
   fQMutex.Lock();
   agemax = fMaxage;
   if ((oldcnt = fCount) > fMininQ) {

      // Prepare to scan down the queue.
      if ((pp = fFirst)) {
         p = pp->fNext;
      } else { p = 0; }

      // Find the first object that's been idle for too long
      while(p && (p->fQTime >= fCurage)) { pp = p; p = p->fNext;}

      // Now delete half of the idle objects. The object queue element must be
      // part of the actual object being deleted for this to properly work.
      if (pp) {
         while (p) {
            pp->fNext = p->fNext;
            delete p->fItem;
            fCount--;
            p = ((pp = pp->fNext) ? pp->fNext : 0);
         }
      }
   }

   // Increase the age and unlock the queue
   fCurage++;
   fQMutex.UnLock();

   // Trace as needed
   if (fTraceON && fTrace->Tracing(fTraceON))
      {fTrace->Beg(fTraceID);
       cerr <<Comment <<" trim done; " <<fCount <<" of " <<oldcnt <<" kept";
       fTrace->End();
      }

   // Reschedule ourselves if we must do so
   if (agemax > 0)
      fSched->Schedule((XrdJob *)this, agemax+time(0));
}

