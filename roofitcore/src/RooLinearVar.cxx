/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooLinearVar.cc,v 1.15 2001/11/19 07:23:56 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
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

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "TObjString.h"
#include "TTree.h"
#include "RooFitCore/RooLinearVar.hh"
#include "RooFitCore/RooStreamParser.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooNumber.hh"

ClassImp(RooLinearVar)

RooLinearVar::RooLinearVar(const char *name, const char *title, RooAbsRealLValue& variable, 
			   const RooAbsReal& slope, const RooAbsReal& offset, const char *unit) :
  RooAbsRealLValue(name, title, unit), 
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
  _var("var",this,other._var),
  _slope("slope",this,other._slope),
  _offset("offset",this,other._offset)
{
  // Copy constructor
}


RooLinearVar::~RooLinearVar() 
{
  // Destructor
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


Double_t RooLinearVar::getFitMin() const 
{
  // Return low end of fit range 
  RooRealVar& var = (RooRealVar&) _var.arg() ;

  if (var.hasFitMin()) {
    if (_slope>0) {
      return _offset + var.getFitMin() * _slope ;
    } else {
      return _offset + var.getFitMax() * _slope ;
    }
  } 
  return -RooNumber::infinity;
}



Double_t RooLinearVar::getFitMax() const 
{
  // Return low end of fit range 
  RooRealVar& var = (RooRealVar&) _var.arg() ;

  if (var.hasFitMax()) {
    if (_slope>0) {
      return _offset + var.getFitMax() * _slope ;
    } else {
      return _offset + var.getFitMin() * _slope ;
    }
  } 
  return +RooNumber::infinity;
}



Int_t RooLinearVar::getFitBins() const 
{
  return ((RooRealVar&)_var.arg()).getFitBins() ;
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
  while (arg=(RooAbsArg*)dIter->Next()) {
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


Bool_t RooLinearVar::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  // Read object contents from stream
  return kTRUE ;
}


void RooLinearVar::writeToStream(ostream& os, Bool_t compact) const
{
  // Write object contents to stream
  if (compact) {
    cout << getVal() ;
  } else {
    cout << _slope.arg().GetName() << " * " << _var.arg().GetName() << " + " << _offset.arg().GetName() ;
  }
}


void RooLinearVar::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this object to the specified stream.

  RooAbsReal::printToStream(os,opt,indent);
  if(opt >= Verbose) {
    os << indent << "--- RooLinearVar ---" << endl;
  }
}

