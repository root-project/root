// @(#)root/base:$Name:  $:$Id: TSysEvtHandler.cxx,v 1.2 2000/11/17 10:26:03 rdm Exp $
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


//______________________________________________________________________________
TFileHandler::TFileHandler(int fd, int mask)
{
   // Create a file descriptor event handler. If mask=kRead then we
   // want to monitor the file for read readiness, if mask=kWrite
   // then we monitor the file for write readiness, if mask=kRead|kWrite
   // then we monitor both read and write readiness.

   fFileNum = fd;
   if (!mask)
      mask = 1;
   fMask = mask;
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


ClassImp(TSignalHandler)

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
