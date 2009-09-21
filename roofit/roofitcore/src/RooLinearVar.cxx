/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
//
// RooLinearVar is the most general form of a derived real-valued object that can
// be used by RooRealIntegral to integrate over. The requirements for this are
//
//          - Can be modified directly (i.e. invertible formula)
//          - Jacobian term in integral is constant (but not necessarily 1)
//
// This class implements the most general form that satisfy these requirement
// 
//    RLV = (slope)*x + (offset)
//
// X is required to be a RooRealVar to meet the invertibility criterium
// (slope) and (offset) is are RooAbsReals, but may not overlap with x,
// i.e. x may not be a server of (slope) and (offset)
//
// In the context of a dataset, (slope) may not contain any real-valued dependents
// (satisfied constant Jacobian requirement). This check cannot be enforced at
// construction time, but can be performed at run time through the isJacobianOK(depList)
// member function.
//
//

#include "RooFit.h"
#include "Riostream.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "TClass.h"
#include "TObjString.h"
#include "TTree.h"
#include "RooLinearVar.h"
#include "RooStreamParser.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooNumber.h"
#include "RooBinning.h"
#include "RooMsgService.h"



ClassImp(RooLinearVar)


//_____________________________________________________________________________
RooLinearVar::RooLinearVar(const char *name, const char *title, RooAbsRealLValue& variable, 
			   const RooAbsReal& slope, const RooAbsReal& offset, const char *unit) :
  RooAbsRealLValue(name, title, unit), 
  _binning(variable.getBinning(),slope.getVal(),offset.getVal()),
  _var("var","variable",this,variable,kTRUE,kTRUE),
  _slope("slope","slope",this,(RooAbsReal&)slope),
  _offset("offset","offset",this,(RooAbsReal&)offset)
{
  // Constructor with RooAbsRealLValue variable and RooAbsReal slope and offset

  // Slope and offset may not depend on variable
  if (slope.dependsOnValue(variable) || offset.dependsOnValue(variable)) {
    coutE(InputArguments) << "RooLinearVar::RooLinearVar(" << GetName() 
			  << "): ERROR, slope(" << slope.GetName() << ") and offset(" 
			  << offset.GetName() << ") may not depend on variable(" 
			  << variable.GetName() << ")" << endl ;
    assert(0) ;
  }

  // Initial plot range and number of bins from dependent variable
//   setPlotRange(variable.getPlotMin()*_slope + _offset,
//                variable.getPlotMax()*_slope + _offset) ;
//   setPlotBins(variable.getPlotBins()) ;
	       
}  



//_____________________________________________________________________________
RooLinearVar::RooLinearVar(const RooLinearVar& other, const char* name) :
  RooAbsRealLValue(other,name), 
  _binning(other._binning),
  _var("var",this,other._var),
  _slope("slope",this,other._slope),
  _offset("offset",this,other._offset)
{
  // Copy constructor
}



//_____________________________________________________________________________
RooLinearVar::~RooLinearVar() 
{
  // Destructor

  _altBinning.Delete() ;
}



//_____________________________________________________________________________
Double_t RooLinearVar::evaluate() const
{
  // Calculate current value of this object  

  return _offset + _var * _slope ;
}



//_____________________________________________________________________________
void RooLinearVar::setVal(Double_t value) 
{
  // Assign given value to linear transformation: sets input variable to (value-offset)/slope
  // If slope is zerom an error message is printed and no assignment is made

  //cout << "RooLinearVar::setVal(" << GetName() << "): new value = " << value << endl ;

  // Prevent DIV0 problems
  if (_slope == 0.) {
    coutE(Eval) << "RooLinearVar::setVal(" << GetName() << "): ERROR: slope is zero, cannot invert relation" << endl ;
    return ;
  }

  // Invert formula 'value = offset + slope*var'
  ((RooRealVar&)_var.arg()).setVal((value - _offset) / _slope) ;
}



//_____________________________________________________________________________
Bool_t RooLinearVar::isJacobianOK(const RooArgSet& depList) const
{
  // Returns true if Jacobian term associated with current
  // expression tree is indeed constant.

  if (!((RooAbsRealLValue&)_var.arg()).isJacobianOK(depList)) {
    return kFALSE ;
  }

  // Check if jacobian has no real-valued dependents
  RooAbsArg* arg ;
  TIterator* dIter = depList.createIterator() ;
  while ((arg=(RooAbsArg*)dIter->Next())) {
    if (arg->IsA()->InheritsFrom(RooAbsReal::Class())) {
      if (_slope.arg().dependsOnValue(*arg)) {
// 	cout << "RooLinearVar::isJacobianOK(" << GetName() << ") return kFALSE because slope depends on value of " << arg->GetName() << endl ;
	return kFALSE ;
      }
    }
  }
  delete dIter ;
//   cout << "RooLinearVar::isJacobianOK(" << GetName() << ") return kTRUE" << endl ;
  return kTRUE ;
}



//_____________________________________________________________________________
Double_t RooLinearVar::jacobian() const 
{
  // Return value of Jacobian associated with the transformation

  return _slope*((RooAbsRealLValue&)_var.arg()).jacobian() ;
}



//_____________________________________________________________________________
Bool_t RooLinearVar::readFromStream(istream& /*is*/, Bool_t /*compact*/, Bool_t /*verbose*/) 
{
  // Read object contents from stream
  return kTRUE ;
}



//_____________________________________________________________________________
void RooLinearVar::writeToStream(ostream& os, Bool_t compact) const
{
  // Write object contents to stream

  if (compact) {
    os << getVal() ;
  } else {
    os << _slope.arg().GetName() << " * " << _var.arg().GetName() << " + " << _offset.arg().GetName() ;
  }
}



//_____________________________________________________________________________
 RooAbsBinning& RooLinearVar::getBinning(const char* name, Bool_t verbose, Bool_t createOnTheFly) 
{
  // Retrieve binning of this linear transformation. A RooLinearVar does not have its own
  // binnings but uses linearly transformed binnings of teh input variable. If a given
  // binning exists on the input variable, it will also exists on this linear transformation
  // and a binning adaptor object is created on the fly.

  // Normalization binning
  if (name==0) {
    _binning.updateInput(((RooAbsRealLValue&)_var.arg()).getBinning(),_slope,_offset) ;
    return _binning ;
  } 

  // Alternative named range binnings, look for existing translator binning first
  RooLinTransBinning* altBinning = (RooLinTransBinning*) _altBinning.FindObject(name) ;
  if (altBinning) {
    altBinning->updateInput(((RooAbsRealLValue&)_var.arg()).getBinning(name,verbose),_slope,_offset) ;
    return *altBinning ;
  }

  // If binning is not found return default binning, if creation is not requested
  if (!createOnTheFly) {
    return _binning ;
  }

  // Create translator binning on the fly
  RooAbsBinning& sourceBinning = ((RooAbsRealLValue&)_var.arg()).getBinning(name,verbose) ;
  RooLinTransBinning* transBinning = new RooLinTransBinning(sourceBinning,_slope,_offset) ;
  _altBinning.Add(transBinning) ;

  return *transBinning ;
}


//_____________________________________________________________________________
const RooAbsBinning& RooLinearVar::getBinning(const char* name, Bool_t verbose, Bool_t createOnTheFly) const
{
  // Const version of getBinning()

  return const_cast<RooLinearVar*>(this)->getBinning(name,verbose,createOnTheFly) ;
}


//_____________________________________________________________________________
Bool_t RooLinearVar::hasBinning(const char* name) const 
{
  // Returns true if binning with given name exists.If a given binning
  // exists on the input variable, it will also exists on this linear
  // transformation.

  return ((RooAbsRealLValue&)_var.arg()).hasBinning(name) ;
}
