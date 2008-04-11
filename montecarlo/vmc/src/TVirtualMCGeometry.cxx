// @(#)root/vmc:$Id$
// Authors: Alice collaboration 25/06/2002

/*************************************************************************
 * Copyright (C) 2006, Rene Brun and Fons Rademakers.                    *
 * Copyright (C) 2002, ALICE Experiment at CERN.                         *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TVirtualMCGeometry.h"

//______________________________________________________________________________
//   Virtual MCGeometry provides a virtual interface to Monte Carlo
//   geometry construction.
//______________________________________________________________________________

ClassImp(TVirtualMCGeometry)

//_____________________________________________________________________________
TVirtualMCGeometry::TVirtualMCGeometry(const char *name, const char *title)
  : TNamed(name,title)
{
   //
   // Standard constructor
   //
}

//_____________________________________________________________________________
TVirtualMCGeometry::TVirtualMCGeometry()
  : TNamed()
{
   //
   // Default constructor
   //
}

//_____________________________________________________________________________
TVirtualMCGeometry::~TVirtualMCGeometry()
{
   //
   // Destructor
   //
}
