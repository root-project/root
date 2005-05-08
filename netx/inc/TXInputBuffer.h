// @(#)root/netx:$Name:  $:$Id: TXInputBuffer.h,v 1.3 2004/12/16 19:23:18 rdm Exp $
// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXInputBuffer
#define ROOT_TXInputBuffer

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXInputBuffer                                                        //
//                                                                      //
// Authors: Alvise Dorigo, Fabrizio Furano                              //
//          INFN Padova, 2003                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TThread
#include "TThread.h"
#endif
#ifndef ROOT_TXMessage
#include "TXMessage.h"
#endif
#include <list>
#include <map>


class TXInputBuffer {

private:

   std::list<TXMessage *>          fMsgQue;      // queue for incoming messages
   TMutex                         *fMutex;       // mutex to protect data structures
   std::map<Short_t, TCondition *> fSyncobjRepo; // each streamid counts on a condition
                                                 // variable to make the caller wait
                                                 // until some data is available
   TCondition     *GetSyncObjOrMakeOne(Short_t streamid);
   Int_t           MsgForStreamidCnt(Short_t streamid);

public:
   TXInputBuffer();
  ~TXInputBuffer();

   inline bool     IsMexEmpty() { return (MexSize() == 0); }
   inline bool     IsSemEmpty() { return (SemSize() == 0); }
   inline long     MexSize() { R__LOCKGUARD(fMutex);
                               return fMsgQue.size();   }
   Int_t           PutMsg(TXMessage *msg);
   inline long     SemSize() { R__LOCKGUARD(fMutex);
                               return fSyncobjRepo.size();  }
   TXMessage      *GetMsg(Short_t streamid, Int_t secstimeout);
   TXMessage      *RetrieveMsg(Short_t streamid);
};

#endif
