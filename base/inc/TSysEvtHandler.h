// @(#)root/base:$Id$
// Author: Fons Rademakers   16/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSysEvtHandler
#define ROOT_TSysEvtHandler


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSysEvtHandler                                                       //
//                                                                      //
// Abstract base class for handling system events.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TQObject
#include "TQObject.h"
#endif


class TSysEvtHandler : public TObject, public TQObject {

private:
   Bool_t   fIsActive;    // kTRUE if handler is active, kFALSE if not active

   void  *GetSender() { return this; }  //used to set gTQSender

public:
   TSysEvtHandler() : fIsActive(kTRUE) { }
   virtual ~TSysEvtHandler() { }

   void             Activate();
   void             DeActivate();
   Bool_t           IsActive() const { return fIsActive; }

   virtual void     Add()    = 0;
   virtual void     Remove() = 0;
   virtual Bool_t   Notify() = 0;

   virtual void     Activated()   { Emit("Activated()"); }   //*SIGNAL*
   virtual void     DeActivated() { Emit("DeActivated()"); } //*SIGNAL*
   virtual void     Notified()    { Emit("Notified()"); }    //*SIGNAL*
   virtual void     Added()       { Emit("Added()"); }       //*SIGNAL*
   virtual void     Removed()     { Emit("Removed()"); }     //*SIGNAL*

   ClassDef(TSysEvtHandler,0)  //ABC for handling system events
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFileHandler                                                         //
//                                                                      //
// Handles events on file descriptors.                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TFileHandler : public TSysEvtHandler {

protected:
   int  fFileNum;     //File descriptor
   int  fMask;        //Event interest mask, either bit 1 (read), 2 (write) or both can be set
   int  fReadyMask;   //Readiness mask, either bit 1 (read), 2 (write) or both can be set

   TFileHandler(): fFileNum(-1), fMask(0), fReadyMask(0) { }

public:
   enum { kRead = 1, kWrite = 2 };

   TFileHandler(int fd, int mask);
   virtual ~TFileHandler() { Remove(); }
   int             GetFd() const { return fFileNum; }
   void            SetFd(int fd) { fFileNum = fd; }
   virtual Bool_t  Notify();
   virtual Bool_t  ReadNotify();
   virtual Bool_t  WriteNotify();
   virtual Bool_t  HasReadInterest();
   virtual Bool_t  HasWriteInterest();
   virtual void    SetInterest(Int_t mask);
   virtual void    ResetReadyMask() { fReadyMask = 0; }
   virtual void    SetReadReady() { fReadyMask |= 0x1; }
   virtual void    SetWriteReady() { fReadyMask |= 0x2; }
   virtual Bool_t  IsReadReady() const { return (fReadyMask & 0x1) == 0x1; }
   virtual Bool_t  IsWriteReady() const { return (fReadyMask & 0x2) == 0x2; }
   virtual void    Add();
   virtual void    Remove();

   ClassDef(TFileHandler,0)  //Handles events on file descriptors
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSignalHandler                                                       //
//                                                                      //
// Handles signals.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

enum ESignals {
   kSigBus,
   kSigSegmentationViolation,
   kSigSystem,
   kSigPipe,
   kSigIllegalInstruction,
   kSigQuit,
   kSigInterrupt,
   kSigWindowChanged,
   kSigAlarm,
   kSigChild,
   kSigUrgent,
   kSigFloatingException,
   kSigTermination,
   kSigUser1,
   kSigUser2
};


class TSignalHandler : public TSysEvtHandler {

protected:
   ESignals    fSignal;   //Signal to be handled
   Bool_t      fSync;     //Synchronous or a-synchronous signal
   Int_t       fDelay;    //Delay handling of signal (use fDelay in Notify())

   TSignalHandler(): fSignal((ESignals)-1), fSync(kTRUE), fDelay(0) { }

public:
   TSignalHandler(ESignals sig, Bool_t sync = kTRUE);
   virtual ~TSignalHandler() { Remove(); }
   void           Delay() { fDelay = 1; }
   void           HandleDelayedSignal();
   ESignals       GetSignal() const { return fSignal; }
   void           SetSignal(ESignals sig) { fSignal = sig; }
   Bool_t         IsSync() const { return fSync; }
   Bool_t         IsAsync() const { return !fSync; }
   virtual Bool_t Notify();
   virtual void   Add();
   virtual void   Remove();

   ClassDef(TSignalHandler,0)  //Signal event handler
};

inline void TSignalHandler::HandleDelayedSignal()
{
   if (fDelay > 1) {
      fDelay = 0;
      Notify();
   } else
      fDelay = 0;
}


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStdExceptionHandler                                                 //
//                                                                      //
// Handles standard C++ exceptions.                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

namespace std { class exception; }

class TStdExceptionHandler : public TSysEvtHandler {

public:
   enum EStatus { kSEProceed, kSEHandled, kSEAbort };

   TStdExceptionHandler();
   virtual ~TStdExceptionHandler() { }

   virtual void     Add();
   virtual void     Remove();
   virtual Bool_t   Notify();

   virtual EStatus  Handle(std::exception& exc) = 0;

   ClassDef(TStdExceptionHandler,0)  //C++ exception handler
};

#endif
