// @(#)root/base:$Id: 8944840ba34631ec28efc779647618db43c0eee5 $
// Author: Fons Rademakers   15/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TSystem

Abstract base class defining a generic interface to the underlying
Operating System.
This is not an ABC in the strict sense of the (C++) word. For
every member function there is an implementation (often not more
than a call to AbstractMethod() which prints a warning saying
that the method should be overridden in a derived class), which
allows a simple partial implementation for new OS'es.
*/

#ifdef WIN32
#include <io.h>
#endif
#include <stdlib.h>
#include <errno.h>
#include <algorithm>
#include <sys/stat.h>

#include "Riostream.h"
#include "TSystem.h"
#include "TSigHandling.h"
#include "TApplication.h"
#include "TException.h"
#include "TROOT.h"
#include "TClass.h"
#include "TClassTable.h"
#include "TEnv.h"
#include "TBrowser.h"
#include "TString.h"
#include "TOrdCollection.h"
#include "TInterpreter.h"
#include "TRegexp.h"
#include "TTimer.h"
#include "TObjString.h"
#include "TError.h"
#include "TPluginManager.h"
#include "TUrl.h"
#include "TVirtualMutex.h"
#include "compiledata.h"
#include "RConfigure.h"

TSigHandling  *gSigHandling = 0;

ClassImp(TSigHandling)

////////////////////////////////////////////////////////////////////////////////
/// Create a new OS interface.

TSigHandling::TSigHandling(const char *name, const char *title) : TNamed(name, title)
{
   fSignals       = 0;
   fSigcnt        = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Delete the OS interface.

TSigHandling::~TSigHandling()
{
   if (fSignalHandler) {
      fSignalHandler->Delete();
      SafeDelete(fSignalHandler);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Init the OS interface.
void TSigHandling::Init()
{
   fSignalHandler = new TOrdCollection;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a signal handler to list of system signal handlers. Only adds
/// the handler if it is not already in the list of signal handlers.

Bool_t TSigHandling::HaveTrappedSignal(Bool_t)
{
   AbstractMethod("HaveTrappedSignal");
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Dispatch signals.
 
void TSigHandling::DispatchSignals(ESignals /*sig*/)
{
   AbstractMethod("DispatchSignals");
}


////////////////////////////////////////////////////////////////////////////////
/// Add a signal handler to list of system signal handlers. Only adds
/// the handler if it is not already in the list of signal handlers.

void TSigHandling::AddSignalHandler(TSignalHandler *)
{
   AbstractMethod("AddSignalHandler");
}

////////////////////////////////////////////////////////////////////////////////
/// Remove a signal handler from list of signal handlers. Returns
/// the handler or 0 if the handler was not in the list of signal handlers.

TSignalHandler *TSigHandling::RemoveSignalHandler(TSignalHandler *)
{
   AbstractMethod("RemoveSignalHandler");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// If reset is true reset the signal handler for the specified signal
/// to the default handler, else restore previous behaviour.

void TSigHandling::ResetSignal(ESignals /*sig*/, Bool_t /*reset*/)
{
   AbstractMethod("ResetSignal");
}

////////////////////////////////////////////////////////////////////////////////
/// Reset signals handlers to previous behaviour.

void TSigHandling::ResetSignals()
{
   AbstractMethod("ResetSignals");
}

////////////////////////////////////////////////////////////////////////////////
/// If ignore is true ignore the specified signal, else restore previous
/// behaviour.

void TSigHandling::IgnoreSignal(ESignals /*sig*/, Bool_t /*ignore*/)
{
   AbstractMethod("IgnoreSignal");
}

////////////////////////////////////////////////////////////////////////////////
/// If ignore is true ignore the interrupt signal, else restore previous
/// behaviour. Typically call ignore interrupt before writing to disk.

void TSigHandling::IgnoreInterrupt(Bool_t ignore)
{
   IgnoreSignal(kSigInterrupt, ignore);
}

////////////////////////////////////////////////////////////////////////////////
/// Obtain the current signal handlers
TSeqCollection *TSigHandling::GetListOfSignalHandlers()
{
   return fSignalHandler;
}

////////////////////////////////////////////////////////////////////////////////
/// Print a stack trace.

void TSigHandling::StackTrace()
{
   AbstractMethod("StackTrace");
}
