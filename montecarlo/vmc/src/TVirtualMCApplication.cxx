// @(#)root/vmc:$Id$
// Author: Ivana Hrivnacova, 27/03/2002

/*************************************************************************
 * Copyright (C) 2006, Rene Brun and Fons Rademakers.                    *
 * Copyright (C) 2002, ALICE Experiment at CERN.                         *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TVirtualMCApplication.h"
#include "TError.h"
#include "TVirtualMC.h"
#include "TMCManager.h"

/** \class TVirtualMCApplication
    \ingroup vmc

Interface to a user Monte Carlo application.

*/

TMCThreadLocal TVirtualMCApplication *TVirtualMCApplication::fgInstance = nullptr;
Bool_t TVirtualMCApplication::fLockMultiThreading = kFALSE;

////////////////////////////////////////////////////////////////////////////////
///
/// Standard constructor
///

TVirtualMCApplication::TVirtualMCApplication(const char *name, const char *title) : TNamed(name, title)
{
   if (fgInstance) {
      ::Fatal("TVirtualMCApplication::TVirtualMCApplication", "Attempt to create two instances of singleton.");
   }

   // This is set to true if a TMCManager was reuqested.
   if (fLockMultiThreading) {
      ::Fatal("TVirtualMCApplication::TVirtualMCApplication", "In multi-engine run ==> multithreading is disabled.");
   }

   fgInstance = this;
   // There cannot be a TVirtualMC since it must have registered to this
   // TVirtualMCApplication
   fMC = nullptr;
   fMCManager = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Default constructor
///

TVirtualMCApplication::TVirtualMCApplication() : TNamed()
{
   fgInstance = this;
   fMC = nullptr;
   fMCManager = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Destructor
///

TVirtualMCApplication::~TVirtualMCApplication()
{
   fgInstance = nullptr;
   if (fMCManager) {
      delete fMCManager;
   }
}

////////////////////////////////////////////////////////////////////////////////
///
/// Static access method
///

TVirtualMCApplication *TVirtualMCApplication::Instance()
{
   return fgInstance;
}

////////////////////////////////////////////////////////////////////////////////
///
/// For backwards compatibility provide a static GetMC method
///

void TVirtualMCApplication::RequestMCManager()
{
   fMCManager = new TMCManager();
   fMCManager->Register(this);
   fMCManager->ConnectEnginePointer(&fMC);
   fLockMultiThreading = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
///
/// /// Register the an engine.
///

void TVirtualMCApplication::Register(TVirtualMC *mc)
{
   // If there is already a transport engine, fail since only one is allowed.
   if (fMC && !fMCManager) {
      Fatal("Register", "Attempt to register a second TVirtualMC which "
                        "is not allowed");
   }
   fMC = mc;
   if (fMCManager) {
      fMCManager->Register(mc);
   }
}

////////////////////////////////////////////////////////////////////////////////
///
/// Return the current transport engine in use
///

TVirtualMC *TVirtualMCApplication::GetMC() const
{
   return fMC;
}
