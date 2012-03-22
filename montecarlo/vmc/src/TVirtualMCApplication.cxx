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

//______________________________________________________________________________
//
// Interface to a user Monte Carlo application.
//______________________________________________________________________________

ClassImp(TVirtualMCApplication)

#if defined(__linux__) && !defined(__CINT__)
__thread TVirtualMCApplication* TVirtualMCApplication::fgInstance = 0;
#else
TVirtualMCApplication* TVirtualMCApplication::fgInstance = 0;
#endif

//_____________________________________________________________________________
TVirtualMCApplication::TVirtualMCApplication(const char *name,
                                             const char *title)
  : TNamed(name,title)
{
//
// Standard constructor
//

   if (fgInstance) {
      Fatal("TVirtualMCApplication",
            "Attempt to create two instances of singleton.");
   }

   fgInstance = this;
}

//_____________________________________________________________________________
TVirtualMCApplication::TVirtualMCApplication()
  : TNamed()
{
   //
   // Default constructor
   //
   fgInstance = this;
}

//_____________________________________________________________________________
TVirtualMCApplication::~TVirtualMCApplication()
{
   //
   // Destructor
   //

   fgInstance = 0;
}

//_____________________________________________________________________________
TVirtualMCApplication* TVirtualMCApplication::Instance()
{
   //
   // Static access method
   //

   return fgInstance;
}

