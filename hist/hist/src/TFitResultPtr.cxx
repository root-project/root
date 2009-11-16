// @(#)root/hist:$Id$
// Author: David Gonzalez Maline   12/11/09

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TFitResultPtr.h"
#include "TFitResult.h"
#include <cassert>

/**
TFitResultPtr provides an indirection to the TFitResult class and with a semantics
identical to a TFitResult pointer, i.e. it is like a smart pointer to a TFitResult. 
In addition it provides an automatic comversion to an integer. In this way it can be 
returned from the TH1::Fit method and the change in TH1::Fit be backward compatible. 
The class 

 */

ClassImp(TFitResultPtr)

TFitResultPtr::TFitResultPtr(const TFitResultPtr& rhs) : 
   fStatus(rhs.fStatus), fPointer(0)
{
   // copy constructor - create a new TFitResult if needed
   if (rhs.fPointer != 0)  fPointer = new TFitResult(*rhs);
}

TFitResultPtr::~TFitResultPtr()
{
   // destructor - delete the contained TFitResult pointer if needed
   if ( fPointer != 0)
      delete fPointer;
}

TFitResultPtr::operator int() const 
{
   // automatic integer conversion. Needed for backward-compatibility with int TH1::FIt
   // The returned integer is the fit status code from FitResult
   if ( fPointer == 0 )
      return fStatus;
   else
      return fPointer->Status();
}

TFitResult& TFitResultPtr::operator*() const
{
   // impelment the de-reference operator to make the class acts as a pointer to a TFitResult
   // assert in case the class does not contain a pointer to TFitResult
   assert (fPointer != 0);
   return *fPointer;
}

TFitResult* TFitResultPtr::operator->() const
{
   // implement the -> operator to make the class acts as a pointer to a TFitResult
   // assert in case the class does not contain a pointer to TFitResult
   assert (fPointer != 0);
   return fPointer;
}


TFitResultPtr & TFitResultPtr::operator=(const TFitResultPtr& rhs) 
{ 
   // assignment operator
   // if needed copy the TFitResult  object and delete previous one if existing
   if ( &rhs == this) return *this; // self assignment
   fStatus = rhs.fStatus; 
   if ( fPointer ) delete fPointer;
   fPointer = 0;
   if (rhs.fPointer != 0)  fPointer = new TFitResult(*rhs);
}

