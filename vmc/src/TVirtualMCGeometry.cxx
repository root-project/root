// @(#)root/vmc:$Name:  $:$Id: TVirtualMCGeometry.cxx,v 1.2 2006/05/13 20:57:20 brun Exp $
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

TVirtualMCGeometry* TVirtualMCGeometry::fgInstance=0;

//_____________________________________________________________________________
TVirtualMCGeometry::TVirtualMCGeometry(const char *name, const char *title)
  : TNamed(name,title)
{
   //
   // Standard constructor
   //
   if (fgInstance)
      Warning("TVirtualMCGeometry","TVirtualMCGeometry instance already exists");
   else
      fgInstance=this;
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
   fgInstance=0;
}
