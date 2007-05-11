/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooLinearVar.cc,v 1.29 2005/06/20 15:44:54 wverkerke Exp $
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

// -- CLASS DESCRIPTION [REAL] --
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
// construction time, but can be performed any time using the isJacobianOK(depList)
// member function.
//

#include "RooFit.h"

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

ClassImp(RooLinearVar)

RooLinearVar::RooLinearVar(const char *name, const char *title, RooAbsRealLValue& variable, 
			   const RooAbsReal& slope, const RooAbsReal& offset, const char *unit) :
  RooAbsRealLValue(name, title, unit), 
  _binning(variable.getBinning(),slope.getVal(),offset.getVal()),
  _var("var","variable",this,variable,kTRUE,kTRUE),
  _slope("slope","slope",this,(RooAbsReal&)slope),
  _offset("offset","offset",this,(RooAbsReal&)offset)
{
  // Constructor with RooRealVar variable and RooAbsReal slope and offset

  // Slope and offset may not depend on variable
  if (slope.dependsOn(variable) || offset.dependsOn(variable)) {
    cout << "RooLinearVar::RooLinearVar(" << GetName() 
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


RooLinearVar::RooLinearVar(const RooLinearVar& other, const char* name) :
  RooAbsRealLValue(other,name), 
  _binning(other._binning),
  _var("var",this,other._var),
  _slope("slope",this,other._slope),
  _offset("offset",this,other._offset)
{
  // Copy constructor
}


RooLinearVar::~RooLinearVar() 
{
  // Destructor
  _altBinning.Delete() ;
}


Double_t RooLinearVar::evaluate() const
{
  // Calculate current value of this object  
  return _offset + _var * _slope ;
}


void RooLinearVar::setVal(Double_t value) 
{
  // Assign given value to linear transformation: set input variable to (value-offset)/slope

  //cout << "RooLinearVar::setVal(" << GetName() << "): new value = " << value << endl ;

  // Prevent DIV0 problems
  if (_slope == 0.) {
    cout << "RooLinearVar::setVal(" << GetName() << "): ERROR: slope is zero, cannot invert relation" << endl ;
    return ;
  }

  // Invert formula 'value = offset + slope*var'
  ((RooRealVar&)_var.arg()).setVal((value - _offset) / _slope) ;
}



Bool_t RooLinearVar::isJacobianOK(const RooArgSet& depList) const
{
  // Check if Jacobian of input LValue is OK
  if (!((RooAbsRealLValue&)_var.arg()).isJacobianOK(depList)) {
    return kFALSE ;
  }

  // Check if jacobian has no real-valued dependents
  RooAbsArg* arg ;
  TIterator* dIter = depList.createIterator() ;
  while ((arg=(RooAbsArg*)dIter->Next())) {
    if (arg->IsA()->InheritsFrom(RooAbsReal::Class())) {
      if (_slope.arg().dependsOn(*arg)) {
	return kFALSE ;
      }
    }
  }
  return kTRUE ;
}


Double_t RooLinearVar::jacobian() const 
{
  return _slope*((RooAbsRealLValue&)_var.arg()).jacobian() ;
}



Bool_t RooLinearVar::readFromStream(istream& /*is*/, Bool_t /*compact*/, Bool_t /*verbose*/) 
{
  // Read object contents from stream
  return kTRUE ;
}


void RooLinearVar::writeToStream(ostream& os, Bool_t compact) const
{
  // Write object contents to stream
  if (compact) {
    os << getVal() ;
  } else {
    os << _slope.arg().GetName() << " * " << _var.arg().GetName() << " + " << _offset.arg().GetName() ;
  }
}


void RooLinearVar::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this object to the specified stream.

  RooAbsReal::printToStream(os,opt,indent);
  if(opt >= Verbose) {
    os << indent << "--- RooLinearVar ---" << endl;
  }
}


 RooAbsBinning& RooLinearVar::getBinning(const char* name, Bool_t verbose, Bool_t createOnTheFly) 
{
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

const RooAbsBinning& RooLinearVar::getBinning(const char* name, Bool_t verbose, Bool_t createOnTheFly) const
{
  return const_cast<RooLinearVar*>(this)->getBinning(name,verbose,createOnTheFly) ;
}

Bool_t RooLinearVar::hasBinning(const char* name) const 
{
  return ((RooAbsRealLValue&)_var.arg()).hasBinning(name) ;
}
