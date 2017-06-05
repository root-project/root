// @(#)root/base:$Id$
// Author: Fons Rademakers   16/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TSysEvtHandler
\ingroup Base

Abstract base class for handling system events.
*/

#include "TSysEvtHandler.h"
#include "TSystem.h"


ClassImp(TSysEvtHandler);
ClassImp(TFileHandler);
ClassImp(TSignalHandler);
ClassImp(TStdExceptionHandler);


////////////////////////////////////////////////////////////////////////////////
/// Activate a system event handler. All handlers are by default
/// activated. Use this method to activate a de-activated handler.

void TSysEvtHandler::Activate()
{
   fIsActive = kTRUE;
   Activated();      // emit Activated() signal
}

////////////////////////////////////////////////////////////////////////////////
/// De-activate a system event handler. Use this method to temporarily
/// disable an event handler to avoid it from being recursively called.
/// Use DeActivate() / Activate() instead of Remove() / Add() for this
/// purpose, since the Add() will add the handler back to the end of
/// the list of handlers and cause it to be called again for the same,
/// already handled, event.

void TSysEvtHandler::DeActivate()
{
   fIsActive = kFALSE;
   DeActivated();    // emit DeActivated() signal
}


////////////////////////////////////////////////////////////////////////////////
/// Create a file descriptor event handler. If mask=kRead then we
/// want to monitor the file for read readiness, if mask=kWrite
/// then we monitor the file for write readiness, if mask=kRead|kWrite
/// then we monitor both read and write readiness.

TFileHandler::TFileHandler(int fd, int mask)
{
   fFileNum = fd;
   if (!mask)
      mask = kRead;
   fMask = mask;
   fReadyMask = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Notify when event occurred on descriptor associated with this handler.

Bool_t TFileHandler::Notify()
{
   Notified();       // emit Notified() signal
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Notify when something can be read from the descriptor associated with
/// this handler.

Bool_t TFileHandler::ReadNotify()
{
   Notified();       // emit Notified() signal
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Notify when something can be written to the descriptor associated with
/// this handler.

Bool_t TFileHandler::WriteNotify()
{
   Notified();       // emit Notified() signal
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// True if handler is interested in read events.

Bool_t TFileHandler::HasReadInterest()
{
   return (fMask & 1);
}

////////////////////////////////////////////////////////////////////////////////
/// True if handler is interested in write events.

Bool_t TFileHandler::HasWriteInterest()
{
   return (fMask & 2);
}

////////////////////////////////////////////////////////////////////////////////
/// Set interest mask to 'mask'.

void TFileHandler::SetInterest(Int_t mask)
{
   if (!mask)
      mask = kRead;
   fMask = mask;
}

////////////////////////////////////////////////////////////////////////////////
/// Add file event handler to system file handler list.

void TFileHandler::Add()
{
   if (gSystem && fFileNum != -1) {
      gSystem->AddFileHandler(this);
      Added();      // emit Added() signal
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Remove file event handler from system file handler list.

void TFileHandler::Remove()
{
   if (gSystem && fFileNum != -1) {
      gSystem->RemoveFileHandler(this);
      Removed();     // emit Removed() signal
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Create signal event handler.

TSignalHandler::TSignalHandler(ESignals sig, Bool_t sync)
{
   fSignal = sig;
   fSync   = sync;
   fDelay  = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Notify when signal occurs.

Bool_t TSignalHandler::Notify()
{
   Notified();       // emit Notified() signal
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Add signal handler to system signal handler list.

void TSignalHandler::Add()
{
   if (gSystem && fSignal != (ESignals)-1) {
      gSystem->AddSignalHandler(this);
      Added();      // emit Added() signal
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Remove signal handler from system signal handler list.

void TSignalHandler::Remove()
{
   if (gSystem && fSignal != (ESignals)-1) {
      gSystem->RemoveSignalHandler(this);
      Removed();     // emit Removed() signal
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Handle standard C++ exceptions intercepted by the TSystem::Run().
///
/// Virtual method EStatus Handle(std::exception& exc) is called on the
/// collection of handlers registered to TSystem. The return value of
/// each handler influences the continuation of handling procedure:
///  - kSEProceed - Proceed with passing of the exception to other
///                 handlers, the exception has not been handled.
///  - kSEHandled - The exception has been handled, do not pass it to
///                 other handlers.
///  - kSEAbort   - Abort application.
/// If all handlers return kSEProceed TSystem::Run() rethrows the
/// exception, possibly resulting in process abortion.

TStdExceptionHandler::TStdExceptionHandler() : TSysEvtHandler()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Add std::exception handler to system handler list.

void TStdExceptionHandler::Add()
{
   if (gSystem) {
      gSystem->AddStdExceptionHandler(this);
      Added();      // emit Added() signal
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Remove std::exception handler from system handler list.

void TStdExceptionHandler::Remove()
{
   if (gSystem) {
      gSystem->RemoveStdExceptionHandler(this);
      Removed();     // emit Removed() signal
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Notify when signal occurs.

Bool_t TStdExceptionHandler::Notify()
{
   Notified();       // emit Notified() signal
   return kFALSE;
}
