// @(#)root/win32gdk:$Id$
// Author: Valeriy Onuchin  08/08/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGWin32ProxyBase
#define ROOT_TGWin32ProxyBase

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

typedef void (*TGWin32CallBack)(void*);
class TList;
class TGWin32ProxyBasePrivate;
////////////////////////////////////////////////////////////////////////////////
class TGWin32ProxyBase {
friend class TGWin32;
friend class TGWin32GL;

protected:
   TGWin32ProxyBasePrivate *fPimpl;       // very private data
   Int_t             fBatchLimit;         // batch limit
   TList             *fListOfCallBacks;   // list of callbacks (used for batch processing)
   TGWin32CallBack   fCallBack;           // callback function (executed by "main" thread)
   void              *fParam;             // arguments passed to/from callback function
   ULong_t           fId;                 // thread id. There is one proxy per client thread
   Bool_t            fIsVirtualX;         // true if actual TVirtualX implementation
   static UInt_t     fMaxResponseTime;    // max period for waiting response from server thread 
   static Long_t     fgLock;              // fgLock=1 - all client threads locked

   virtual Bool_t ForwardCallBack(Bool_t sync);
   virtual void   SendExitMessage();

public: // private:
   static ULong_t    fgPostMessageId;     // post message ID
   static ULong_t    fgPingMessageId;     // ping message ID
   static ULong_t    fgMainThreadId;      // main thread ID
   static ULong_t    fgUserThreadId;      // user (e.g. python) thread ID

   static void    Lock();
   static void    Unlock();
   static void    GlobalLock();
   static void    GlobalUnlock();
   static Bool_t  IsGloballyLocked();
   static Bool_t  Ping();

public:
   TGWin32ProxyBase();
   virtual ~TGWin32ProxyBase();
   virtual void      ExecuteCallBack(Bool_t sync);
   virtual Double_t  GetMilliSeconds();
   ULong_t GetId() const { return fId; }
};

#endif
