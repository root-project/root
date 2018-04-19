// @(#)root/vmc:$Id$
// Authors: Ivana Hrivnacova 19/04/2018

/*************************************************************************
 * Copyright (C) 2006, Rene Brun and Fons Rademakers.                    *
 * Copyright (C) 2018, ALICE Experiment at CERN.                         *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TVirtualMCSensitiveDetector.h"

/** \class TVirtualMCSensitiveDetector
    \ingroup vmc

Interface to a user defined sensitive detector.
*/

ClassImp(TVirtualMCSensitiveDetector);

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor

TVirtualMCSensitiveDetector::TVirtualMCSensitiveDetector(const char *name, const char *title) : TNamed(name, title) {}

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor

TVirtualMCSensitiveDetector::TVirtualMCSensitiveDetector(const TString &name, const TString &title)
   : TNamed(name, title)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TVirtualMCSensitiveDetector::TVirtualMCSensitiveDetector() : TNamed() {}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

TVirtualMCSensitiveDetector::TVirtualMCSensitiveDetector(const TVirtualMCSensitiveDetector &rhs) : TNamed(rhs)
{
   /// Copy constructor

   *this = rhs;
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

TVirtualMCSensitiveDetector::~TVirtualMCSensitiveDetector() {}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator

TVirtualMCSensitiveDetector &TVirtualMCSensitiveDetector::operator=(const TVirtualMCSensitiveDetector &rhs)
{
   // check assignment to self
   if (this == &rhs)
      return *this;

   // base class assignment
   TNamed::operator=(rhs);

   return *this;
}
