// @(#)root/win32gdk:$Id$
// Author: Valeriy Onuchin  08/08/2003

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TGWin32ProxyBase
\ingroup win32

Proxy classes provide thread-safe interface to global objects.

For example: TGWin32VirtualXProxy (to gVirtualX).

Proxy object creates callback object and posts a windows message to
"processing thread". When windows message is received callback
("real method") is executed.

For example:
   gVirtualX->ClearWindow()

   - callback object created containing pointer to function
     corresponding TGWin32::ClearWindow() method
   - message to "processing thread" (main thread) is posted
   - TGWin32::ClearWindow() method is executed inside main thread
   - thread containing gVirtualX proxy object waits for reply
     from main thread that TGWin32::ClearWindow() is completed.

Howto create proxy class:

 1. Naming.
      name of proxy = TGWin32 + the name of "virtual base class" + Proxy

      e.g. TGWin32VirtualXProxy = TGWin32 + VirtualX + Proxy

 2. Definition of global object
      As example check definition and implementation of
      gVirtualX global object

 3. Class definition.
      proxy class must be inherited from "virtual base class" and
      TGWin32ProxyBase class. For example:

      class TGWin32VirtualX : public TVirtualX , public  TGWin32ProxyBase

 4. Constructors, destructor, extra methods.
    - constructors and destructor of proxy class do nothing
    - proxy class must contain two extra static methods
      RealObject(), ProxyObject(). Each of them return pointer to object
      of virtual base class.

    For example:
      static TVirtualX *RealObject();
      static TVirtualX *ProxyObject();

 5. Implementation
      TGWin32ProxyDefs.h file contains a set of macros which very
      simplify implementation.
    - RETURN_PROXY_OBJECT macro implements ProxyObject() method, e.g.
      RETURN_PROXY_OBJECT(VirtualX)
    - the names of other macros say about itself.

      For example:
         VOID_METHOD_ARG0(VirtualX,SetFillAttributes,1)
            void TGWin32VirtualXProxy::SetFillAttributes()

         RETURN_METHOD_ARG0_CONST(VirtualX,Visual_t,GetVisual)
            Visual_t TGWin32VirtualXProxy::GetVisual() const

         RETURN_METHOD_ARG2(VirtualX,Int_t,OpenPixmap,UInt_t,w,UInt_t,h)
            Int_t TGWin32VirtualXProxy::OpenPixmap,UInt_t w,UInt_t h)

*/


#include "Windows4Root.h"
#include <windows.h>

#include "TGWin32ProxyBase.h"
#include "TRefCnt.h"
#include "TList.h"
#include "TGWin32.h"
#include "TROOT.h"

////////////////////////////////////////////////////////////////////////////////
class TGWin32CallBackObject : public TObject {
public:
   TGWin32CallBack   fCallBack;  // callback function (called by GUI thread)
   void              *fParam;    // arguments passed to/from callback function

   TGWin32CallBackObject(TGWin32CallBack cb,void *p):fCallBack(cb),fParam(p) {}
   ~TGWin32CallBackObject() { if (fParam) delete fParam; }
};

////////////////////////////////////////////////////////////////////////////////
class TGWin32ProxyBasePrivate {
public:
   HANDLE   fEvent;   // event used for syncronization
   TGWin32ProxyBasePrivate();
   ~TGWin32ProxyBasePrivate();
};

////////////////////////////////////////////////////////////////////////////////
/// ctor

TGWin32ProxyBasePrivate::TGWin32ProxyBasePrivate()
{
   fEvent = ::CreateEvent(NULL, TRUE, FALSE, NULL);
}

////////////////////////////////////////////////////////////////////////////////
/// dtor

TGWin32ProxyBasePrivate::~TGWin32ProxyBasePrivate()
{
   if (fEvent) ::CloseHandle(fEvent);
   fEvent = 0;
}


ULong_t TGWin32ProxyBase::fgPostMessageId = 0;
ULong_t TGWin32ProxyBase::fgPingMessageId = 0;
ULong_t TGWin32ProxyBase::fgMainThreadId = 0;
ULong_t TGWin32ProxyBase::fgUserThreadId = 0;
Long_t  TGWin32ProxyBase::fgLock = 0;
UInt_t  TGWin32ProxyBase::fMaxResponseTime = 0;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/// ctor

TGWin32ProxyBase::TGWin32ProxyBase()
{
   fIsVirtualX = kFALSE;
   fCallBack = 0;
   fParam = 0;
   fListOfCallBacks = new TList();
   fBatchLimit = 100;
   fId = ::GetCurrentThreadId();
   fPimpl = new TGWin32ProxyBasePrivate();

   if (!fgPostMessageId) fgPostMessageId = ::RegisterWindowMessage("TGWin32ProxyBase::Post");
   if (!fgPingMessageId) fgPingMessageId = ::RegisterWindowMessage("TGWin32ProxyBase::Ping");
}

////////////////////////////////////////////////////////////////////////////////
/// dtor

TGWin32ProxyBase::~TGWin32ProxyBase()
{
   fListOfCallBacks->Delete();
   delete fListOfCallBacks;
   fListOfCallBacks = 0;

   delete fPimpl;
}

////////////////////////////////////////////////////////////////////////////////
/// enter critical section

void TGWin32ProxyBase::Lock()
{
   TGWin32::Lock();
}

////////////////////////////////////////////////////////////////////////////////
/// leave critical section

void TGWin32ProxyBase::Unlock()
{
   TGWin32::Unlock();
}

////////////////////////////////////////////////////////////////////////////////
/// lock any proxy (client thread)

void TGWin32ProxyBase::GlobalLock()
{
   if (IsGloballyLocked()) return;
   ::InterlockedIncrement(&fgLock);
}

////////////////////////////////////////////////////////////////////////////////
///  unlock any proxy (client thread)

void TGWin32ProxyBase::GlobalUnlock()
{
   if (!IsGloballyLocked()) return;
   ::InterlockedDecrement(&fgLock);
}

////////////////////////////////////////////////////////////////////////////////
/// send ping messsage to server thread

Bool_t TGWin32ProxyBase::Ping()
{
   return ::PostThreadMessage(fgMainThreadId, fgPingMessageId, (WPARAM)0, 0L);
}

////////////////////////////////////////////////////////////////////////////////
/// returns elapsed time in milliseconds with microseconds precision

Double_t TGWin32ProxyBase::GetMilliSeconds()
{
   static LARGE_INTEGER freq;
   static Bool_t first = kTRUE;
   LARGE_INTEGER count;
   static Double_t overhead = 0;

   if (first) {
      LARGE_INTEGER count0;
      ::QueryPerformanceFrequency(&freq);
      ::QueryPerformanceCounter(&count0);
      if (1) {
         Double_t dummy;
         dummy = ((Double_t)count0.QuadPart - overhead)*1000./((Double_t)freq.QuadPart);
      }
      ::QueryPerformanceCounter(&count);
      overhead = (Double_t)count.QuadPart - (Double_t)count0.QuadPart;
      first = kFALSE;
   }

   ::QueryPerformanceCounter(&count);
   return ((Double_t)count.QuadPart - overhead)*1000./((Double_t)freq.QuadPart);
}

////////////////////////////////////////////////////////////////////////////////
/// Executes all batched callbacks and the latest callback
/// This method is executed by server thread

void TGWin32ProxyBase::ExecuteCallBack(Bool_t sync)
{
   // process batched callbacks
   if (fListOfCallBacks && fListOfCallBacks->GetSize()) {
      TIter next(fListOfCallBacks);
      TGWin32CallBackObject *obj;

      while ((obj = (TGWin32CallBackObject*)next())) {
         obj->fCallBack(obj->fParam);  // execute callback
      }
   }
   if (sync) {
      if (fCallBack) fCallBack(fParam);
      ::SetEvent(fPimpl->fEvent);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// if sync is kTRUE:
///    - post message to main thread.
///    - execute callbacks from fListOfCallBacks
///    - wait for response
/// else
///    -  add callback to fListOfCallBacks
///
/// returns kTRUE if callback execution is delayed (batched)

Bool_t TGWin32ProxyBase::ForwardCallBack(Bool_t sync)
{
   Int_t wait = 0;

   if (!fgMainThreadId) return kFALSE;

   while (IsGloballyLocked()) {
      Ping();
#ifdef OLD_THREAD_IMPLEMENTATION
      if (GetCurrentThreadId() == fgMainThreadId)
         break;
#endif
      ::SleepEx(10, 1); // take a rest
      if (!fgMainThreadId) return kFALSE; // server thread terminated
   }

   Bool_t batch = !sync && (fListOfCallBacks->GetSize() < fBatchLimit);

   // if it is a call to gVirtualX and comes from a secondary thread,
   // delay it and process it via the main thread (to avoid deadlocks).
   if (!fgUserThreadId && fIsVirtualX &&
       (GetCurrentThreadId() != fgMainThreadId) &&
       (fListOfCallBacks->GetSize() < fBatchLimit))
      batch = kTRUE;

   if (batch) {
      fListOfCallBacks->Add(new TGWin32CallBackObject(fCallBack, fParam));
      return kTRUE;
   }

   while (!::PostThreadMessage(fgMainThreadId, fgPostMessageId, (WPARAM)this, 0L)) {
      // wait because there is a chance that message queue does not exist yet
      ::SleepEx(50, 1);
      if (wait++ > 5) return kFALSE; // failed to post
   }

#ifdef OLD_THREAD_IMPLEMENTATION
   Int_t cnt = 0; //VO attempt counters
#endif
   // limiting wait time
   DWORD res = WAIT_TIMEOUT;
   while (res ==  WAIT_TIMEOUT) {
      res = ::WaitForSingleObject(fPimpl->fEvent, 100);
#ifdef OLD_THREAD_IMPLEMENTATION
      if ((GetCurrentThreadId() == fgMainThreadId) ||
         (!gROOT->IsLineProcessing() && IsGloballyLocked())) {
         break;
      }
      if (cnt++ > 20) break; // VO after some efforts go out from loop
#endif
   }
   ::ResetEvent(fPimpl->fEvent);

   if (res == WAIT_TIMEOUT) { // server thread is blocked
      GlobalLock();
      return kTRUE;
   }

   fListOfCallBacks->Delete();
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Check the status of the lock.

Bool_t TGWin32ProxyBase::IsGloballyLocked()
{
   return fgLock;
}

////////////////////////////////////////////////////////////////////////////////
/// send exit message to server thread

void TGWin32ProxyBase::SendExitMessage()
{
   ::PostThreadMessage(fgMainThreadId, WM_QUIT, 0, 0L);
}

