// @(#)root/base:$Id$
// Author: Fons Rademakers   16/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSysEvtHandler                                                       //
//                                                                      //
// Abstract base class for handling system events.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSysEvtHandler.h"
#include "TSystem.h"


ClassImp(TSysEvtHandler)
ClassImp(TFileHandler)
ClassImp(TSignalHandler)
ClassImp(TStdExceptionHandler)


//______________________________________________________________________________
void TSysEvtHandler::Activate()
{
   // Activate a system event handler. All handlers are by default
   // activated. Use this method to activate a de-activated handler.

   fIsActive = kTRUE;
   Activated();      // emit Activated() signal
}

//______________________________________________________________________________
void TSysEvtHandler::DeActivate()
{
   // De-activate a system event handler. Use this method to temporarily
   // disable an event handler to avoid it from being recursively called.
   // Use DeActivate() / Activate() instead of Remove() / Add() for this
   // purpose, since the Add() will add the handler back to the end of
   // the list of handlers and cause it to be called again for the same,
   // already handled, event.

   fIsActive = kFALSE;
   DeActivated();    // emit DeActivated() signal
}


//______________________________________________________________________________
TFileHandler::TFileHandler(int fd, int mask)
{
   // Create a file descriptor event handler. If mask=kRead then we
   // want to monitor the file for read readiness, if mask=kWrite
   // then we monitor the file for write readiness, if mask=kRead|kWrite
   // then we monitor both read and write readiness.

   fFileNum = fd;
   if (!mask)
      mask = kRead;
   fMask = mask;
   fReadyMask = 0;
}

//______________________________________________________________________________
Bool_t TFileHandler::Notify()
{
   // Notify when event occured on descriptor associated with this handler.

   Notified();       // emit Notified() signal
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TFileHandler::ReadNotify()
{
   // Notify when something can be read from the descriptor associated with
   // this handler.

   Notified();       // emit Notified() signal
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TFileHandler::WriteNotify()
{
   // Notify when something can be written to the descriptor associated with
   // this handler.

   Notified();       // emit Notified() signal
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TFileHandler::HasReadInterest()
{
   // True if handler is interested in read events.

   return (fMask & 1);
}

//______________________________________________________________________________
Bool_t TFileHandler::HasWriteInterest()
{
   // True if handler is interested in write events.

   return (fMask & 2);
}

//______________________________________________________________________________
void TFileHandler::SetInterest(Int_t mask)
{
   // Set interest mask to 'mask'.

   if (!mask)
      mask = kRead;
   fMask = mask;
}

//______________________________________________________________________________
void TFileHandler::Add()
{
   // Add file event handler to system file handler list.

   if (gSystem && fFileNum != -1) {
      gSystem->AddFileHandler(this);
      Added();      // emit Added() signal
   }
}

//______________________________________________________________________________
void TFileHandler::Remove()
{
   // Remove file event handler from system file handler list.

   if (gSystem && fFileNum != -1) {
      gSystem->RemoveFileHandler(this);
      Removed();     // emit Removed() signal
   }
}


//______________________________________________________________________________
TSignalHandler::TSignalHandler(ESignals sig, Bool_t sync)
{
   // Create signal event handler.

   fSignal = sig;
   fSync   = sync;
   fDelay  = 0;
}

//______________________________________________________________________________
Bool_t TSignalHandler::Notify()
{
   // Notify when signal occurs.

   Notified();       // emit Notified() signal
   return kFALSE;
}

//______________________________________________________________________________
void TSignalHandler::Add()
{
   // Add signal handler to system signal handler list.

   if (gSystem && fSignal != (ESignals)-1) {
      gSystem->AddSignalHandler(this);
      Added();      // emit Added() signal
   }
}

//______________________________________________________________________________
void TSignalHandler::Remove()
{
   // Remove signal handler from system signal handler list.

   if (gSystem && fSignal != (ESignals)-1) {
      gSystem->RemoveSignalHandler(this);
      Removed();     // emit Removed() signal
   }
}


//______________________________________________________________________________
TStdExceptionHandler::TStdExceptionHandler() : TSysEvtHandler()
{
   // Handle standard C++ exceptions intercepted by the TSystem::Run().
   //
   // Virtual method EStatus Handle(std::exception& exc) is called on the
   // collection of handlers registered to TSystem. The return value of
   // each handler influences the continuation of handling procedure:
   //    kSEProceed - Proceed with passing of the exception to other
   //                 handlers, the exception has not been handled.
   //    kSEHandled - The exception has been handled, do not pass it to
   //                 other handlers.
   //    kSEAbort   - Abort application.
   // If all handlers return kSEProceed TSystem::Run() rethrows the
   // exception, possibly resulting in process abortion.

}

//______________________________________________________________________________
void TStdExceptionHandler::Add()
{
   // Add std::exception handler to system handler list.

   if (gSystem) {
      gSystem->AddStdExceptionHandler(this);
      Added();      // emit Added() signal
   }
}

//______________________________________________________________________________
void TStdExceptionHandler::Remove()
{
   // Remove std::exception handler from system handler list.

   if (gSystem) {
      gSystem->RemoveStdExceptionHandler(this);
      Removed();     // emit Removed() signal
   }
}

//______________________________________________________________________________
Bool_t TStdExceptionHandler::Notify()
{
   // Notify when signal occurs.

   Notified();       // emit Notified() signal
   return kFALSE;
}
