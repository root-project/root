// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
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

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TThread
#include "TThread.h"
#endif
#ifndef ROOT_TXMessage
#include "TXMessage.h"
#endif
#include <list>
#include <map>

using namespace std;

class TXInputBuffer {

private:

   list<TXMessage*>            fMsgQue;      // queue for incoming messages
   list<TXMessage*>::iterator  fMsgIter;     // its iterator
   TMutex                     *fMutex;       // mutex to protect data structures
   map<Short_t, TCondition *>  fSyncobjRepo; // each streamid counts on a condition
                                             // variable to make the caller wait
                                             // until some data is available

   TCondition     *GetSyncObjOrMakeOne(Short_t streamid);
   Int_t           MsgForStreamidCnt(Short_t streamid);

public:
   TXInputBuffer();
  ~TXInputBuffer();

   inline bool     IsMexEmpty() { return (MexSize() == 0); }
   inline bool     IsSemEmpty() { return (SemSize() == 0); }
   inline long     MexSize() { return fMsgQue.size(); }
   Int_t           PutMsg(TXMessage *msg);
   inline long     SemSize() { return fSyncobjRepo.size(); }
   TXMessage      *GetMsg(Short_t streamid, Int_t secstimeout);
};



#endif
