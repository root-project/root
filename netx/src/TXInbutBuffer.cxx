// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXInputBuffer                                                        //
//                                                                      //
// Authors: Alvise Dorigo, Fabrizio Furano                              //
//          INFN Padova, 2003                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TXInputBuffer.h"
#include "TXMutexLocker.h"
#include "TError.h"
#include <map>

using namespace std;

//________________________________________________________________________
Int_t TXInputBuffer::MsgForStreamidCnt(Short_t streamid)
{
    // Counts the number of messages belonging to the given streamid

    Int_t cnt = 0;
    TXMessage *m = 0;

    for (fMsgIter = fMsgQue.begin(); fMsgIter != fMsgQue.end(); ++fMsgIter) {
       m = *fMsgIter;
       if (m->MatchStreamid(streamid))
          cnt++;
    }

    return cnt;
}

//________________________________________________________________________
TCondition *TXInputBuffer::GetSyncObjOrMakeOne(Short_t streamid)
{
   // Gets the right sync obj to wait for messages for a given streamid
   // If the semaphore is not available, it creates one.

   TCondition *cnd;
   map<Short_t, TCondition *>::iterator iter;

   {
      TXMutexLocker mtx(fMutex);
      iter = fSyncobjRepo.find(streamid);
      if (iter == fSyncobjRepo.end()) {
         cnd = new TCondition();
         if (!cnd) {
	    Error("TXInputBuffer::GetSyncObjOrMakeOne",
	          "Fatal ERROR *** Object creation with new failed !"
                  " Probable system resources exhausted.");
	    gSystem->Abort();
         }
         fSyncobjRepo[ streamid ] = cnd;
      } else
         cnd = iter->second;
   }

  return cnd;
}



//_______________________________________________________________________
TXInputBuffer::TXInputBuffer()
{
   // Constructor

   fMutex = new TMutex(kTRUE);
   fMsgQue.clear();
}

//_______________________________________________________________________
TXInputBuffer::~TXInputBuffer()
{
   // Destructor

   SafeDelete(fMutex);
}

//_______________________________________________________________________
Int_t TXInputBuffer::PutMsg(TXMessage* m)
{
   // Put message in the list

   TCondition *cnd;

   {
      TXMutexLocker mtx(fMutex);
    
      fMsgQue.push_back(m);
    
      // Is anybody sleeping ?
      cnd = GetSyncObjOrMakeOne( m->HeaderSID() );
   }

   cnd->Signal();
 
   return MexSize();
}


//_______________________________________________________________________
TXMessage *TXInputBuffer::GetMsg(Short_t streamid, Int_t secstimeout)
{
   // Gets the first TXMessage from the queue, given a matching streamid.
   // If there are no TXMessages for the streamid, it waits for a number
   // of seconds

   TCondition *cv;
   TXMessage *res, *m;
   Int_t cond_ret;

   res = 0;
 
   {
      TXMutexLocker mtx(fMutex);

      if (MsgForStreamidCnt(streamid) > 0) {

         // If there are messages to dequeue, we pick the oldest one
         for (fMsgIter = fMsgQue.begin(); fMsgIter != fMsgQue.end(); ++fMsgIter) {
            m = *fMsgIter;
            if (m->MatchStreamid(streamid)) {
               res = *fMsgIter;
	       fMsgQue.erase(fMsgIter);
	       break;
            }
         }
      } 
   }

   if (!res) {
      {
         TXMutexLocker mtx(fMutex);
         cv = GetSyncObjOrMakeOne(streamid);
      }

      // Remember, the wait primitive internally unlocks the mutex!
      cond_ret = cv->TimedWait(time(0) + secstimeout, 0);
      {
         TXMutexLocker mtx(fMutex);
         // If there are messages to dequeue, we pick the oldest one
         for (fMsgIter = fMsgQue.begin(); fMsgIter != fMsgQue.end(); ++fMsgIter) {
            m = *fMsgIter;
            if (m->MatchStreamid(streamid)) {
	       res = *fMsgIter;
               fMsgQue.erase(fMsgIter);
	       break;
            }
         }
      }
  }

  return res;
}
