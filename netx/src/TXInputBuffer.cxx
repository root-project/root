// @(#)root/netx:$Name:  $:$Id: TXInputBuffer.cxx,v 1.3 2005/01/05 01:20:11 rdm Exp $
// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXInputBuffer                                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <sys/time.h>
#include <stdio.h>
#include "TXInputBuffer.h"
#include "TEnv.h"
#include "TError.h"

using namespace std;

//________________________________________________________________________
Int_t TXInputBuffer::MsgForStreamidCnt(Short_t streamid)
{
    // Counts the number of messages belonging to the given streamid

    Int_t cnt = 0;
    TXMessage *m = 0;

    list<TXMessage *>::iterator i;
    for (i = fMsgQue.begin(); i != fMsgQue.end(); ++i) {
       m = *i;
       if (m && m->MatchStreamid(streamid))
          cnt++;
    }

    return cnt;
}

//________________________________________________________________________
TCondition *TXInputBuffer::GetSyncObjOrMakeOne(Short_t streamid)
{
   // Gets the right sync obj to wait for messages for a given streamid
   // If the semaphore is not available, it creates one.

   TCondition *cnd = 0;
   map<Short_t, TCondition *>::iterator iter;

   {
      R__LOCKGUARD(fMutex);
      iter = fSyncobjRepo.find(streamid);
      if (iter == fSyncobjRepo.end()) {
         cnd = new TCondition(fMutex);
         if (!cnd) {
	    Error("TXInputBuffer::GetSyncObjOrMakeOne",
	          "Fatal ERROR *** Object creation with new failed !"
                  " Probable system resources exhausted.");
	    gSystem->Abort();
         }
         fSyncobjRepo[ streamid ] = cnd;
      } else {
         cnd = iter->second;
      }

   }
   return cnd;
}

//_______________________________________________________________________
TXInputBuffer::TXInputBuffer()
{
   // Constructor

   // Initialization of data structures mutex
   if (!(fMutex = new TMutex(kTRUE)))
      Error("TXInputBuffer", "can't create mutex for data"
                               " structures: out of system resources");
   // Reset queue
   fMsgQue.clear();
}

//_______________________________________________________________________
TXInputBuffer::~TXInputBuffer()
{
   // Destructor

   // Delete all the syncobjs
   {
      R__LOCKGUARD(fMutex);
      fSyncobjRepo.clear();
   }

   SafeDelete(fMutex);
}

//_______________________________________________________________________
Int_t TXInputBuffer::PutMsg(TXMessage* m)
{
   // Put message in the list

   TCondition *cnd = 0;
   Int_t sz = 0, sid = 0;

   {
      R__LOCKGUARD(fMutex);
      fMsgQue.push_back(m);
      sz = MexSize();

      // Is anybody sleeping ?
      if (m)
         cnd = GetSyncObjOrMakeOne( m->HeaderSID() );

      if (m)
          sid = m->HeaderSID();
   }

   // Now we signal to whoever is waiting that we added a new one
   if (m && cnd)
      cnd->Signal();

   return sz;
}

//_______________________________________________________________________
TXMessage *TXInputBuffer::GetMsg(Short_t streamid, Int_t secstimeout)
{
   // Gets the first TXMessage from the queue, given a matching streamid.
   // If there are no TXMessages for the streamid, it waits for a number
   // of seconds

   TCondition *cv = 0;
   TXMessage *res = 0;

   res = RetrieveMsg(streamid);

   if (!res) {

      // Find the cond where to wait for a msg
      cv = GetSyncObjOrMakeOne(streamid);

      for (int k = 0; k < secstimeout; k++) {

         Int_t cr = 0;

         // We have to lock the mtx before going to wait,
         // to avoid missing a signal from PutMsg.
         {
            R__LOCKGUARD(fMutex);

            // Check whether any message arrived in the meantime.
            res = RetrieveMsg(streamid);
            // If not, wait for the next (remember: the mtx is
            // unlocked internally).
            if (!res) {
               cr = cv->TimedWaitRelative(2);
            }
         }

         // If we found something we are done
         if (res) break;

         // If we have been awakened, there might be something
         if (!cr)
            res = RetrieveMsg(streamid);

         // If we found something we are done
         if (res) break;
      }
   }

   return res;
}

//_______________________________________________________________________
TXMessage *TXInputBuffer::RetrieveMsg(Short_t streamid)
{
   // Gets the first TXMessage from the queue, given a matching streamid.
   // If there are no TXMessages for the streamid, it waits for a number
   // of seconds

   TXMessage *res = 0, *m = 0;

   R__LOCKGUARD(fMutex);

   // If there are messages to dequeue, we pick the oldest one
   list<TXMessage *>::iterator i;
   for (i = fMsgQue.begin(); i != fMsgQue.end(); ++i) {
      m = *i;
      if (m && m->MatchStreamid(streamid)) {
         res = m;
         fMsgQue.erase(i);
         break;
      }
   }

   return res;
}
