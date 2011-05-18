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

#include "TVirtualMCStack.h"

//______________________________________________________________________________
//
// Interface to a user defined particles stack.
//______________________________________________________________________________

ClassImp(TVirtualMCStack)

//_____________________________________________________________________________
TVirtualMCStack::TVirtualMCStack()
  : TObject()
{}

//_____________________________________________________________________________
TVirtualMCStack::~TVirtualMCStack()
{}
