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

/** \class TVirtualMCStack
    \ingroup vmc

Interface to a user defined particles stack.
*/

ClassImp(TVirtualMCStack)

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TVirtualMCStack::TVirtualMCStack()
  : TObject()
{}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

TVirtualMCStack::~TVirtualMCStack()
{}
