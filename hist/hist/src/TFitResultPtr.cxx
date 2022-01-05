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
#include "TError.h"

/** \class TFitResultPtr
Provides an indirection to the TFitResult class and with a semantics
identical to a TFitResult pointer, i.e. it is like a smart pointer to a TFitResult.
In addition it provides an automatic conversion to an integer. In this way it can be
returned from the TH1::Fit method and the change in TH1::Fit be backward compatible.
 */

ClassImp(TFitResultPtr);

////////////////////////////////////////////////////////////////////////////////
/// Constructor from a TFitResult pointer

TFitResultPtr::TFitResultPtr(const std::shared_ptr<TFitResult> & p) :
   fStatus(-1),
   fPointer(p)
{
   if (fPointer) fStatus = fPointer->Status();
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor from a TFitResult pointer

TFitResultPtr::TFitResultPtr(TFitResult * p) :
   fStatus(-1),
   fPointer(std::shared_ptr<TFitResult>(p))
{
   if (fPointer) fStatus = fPointer->Status();
}

TFitResultPtr::TFitResultPtr(const TFitResultPtr& rhs) :
   fStatus(rhs.fStatus), fPointer(rhs.fPointer)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor. Delete the contained TFitResult pointer if needed
///     if ( fPointer != 0)
///       delete fPointer;

TFitResultPtr::~TFitResultPtr()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Implement the de-reference operator to make the class acts as a pointer to a TFitResult
/// assert in case the class does not contain a pointer to TFitResult

TFitResult& TFitResultPtr::operator*() const
{
   if  (!fPointer) {
      Error("TFitResultPtr","TFitResult is empty - use the fit option S");
   }
   return *fPointer;
}

////////////////////////////////////////////////////////////////////////////////
/// Implement the -> operator to make the class acts as a pointer to a TFitResult.
/// assert in case the class does not contain a pointer to TFitResult

TFitResult* TFitResultPtr::operator->() const
{
   if  (!fPointer) {
      Error("TFitResultPtr","TFitResult is empty - use the fit option S");
   }
   return fPointer.get();
}

////////////////////////////////////////////////////////////////////////////////
/// Return contained pointer

TFitResult * TFitResultPtr::Get() const {
   return fPointer.get();
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator.
/// if needed copy the TFitResult  object and delete previous one if existing

TFitResultPtr & TFitResultPtr::operator=(const TFitResultPtr& rhs)
{
   if ( &rhs == this) return *this; // self assignment
   fStatus = rhs.fStatus;
   fPointer = rhs.fPointer; 
   // if ( fPointer ) delete fPointer;
   // fPointer = 0;
   // if (rhs.fPointer != 0)  fPointer = new TFitResult(*rhs);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Print the TFitResultPtr by printing its TFitResult.

std::string cling::printValue(const TFitResultPtr* val) {
   if (TFitResult* fr = val->Get())
      return printValue(fr);
   return "<nullptr TFitResult>";
}
