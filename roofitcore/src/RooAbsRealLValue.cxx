/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsRealLValue.cc,v 1.3 2001/05/11 23:37:40 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "TObjString.h"
#include "TTree.h"
#include "RooFitCore/RooAbsRealLValue.hh"
#include "RooFitCore/RooStreamParser.hh"

ClassImp(RooAbsRealLValue)

RooAbsRealLValue::RooAbsRealLValue(const char *name, const char *title, const char *unit) :
  RooAbsReal(name, title, 0, 0, unit)
{
}  


RooAbsRealLValue::RooAbsRealLValue(const RooAbsRealLValue& other, const char* name) :
  RooAbsReal(other,name)
{
}


RooAbsRealLValue::~RooAbsRealLValue() 
{
}



Bool_t RooAbsRealLValue::inFitRange(Double_t value, Double_t* clippedValPtr) const
{
  // Return kTRUE if the input value is within our fit range. Otherwise, return
  // kFALSE and write a clipped value into clippedValPtr if it is non-zero.

  Double_t range = getFitMax() - getFitMin() ; // ok for +/-INIFINITY
  Double_t clippedValue(value);
  Bool_t inRange(kTRUE) ;

  // test this value against our upper fit limit
  if(hasFitMax() && value > getFitMax()) {
    if(value - getFitMax() > 1e-6*range) {
      if (clippedValPtr)
	cout << "RooAbsRealLValue::inFitRange(" << GetName() << "): value " << value
	     << " rounded down to max limit " << getFitMax() << endl;
    }
    clippedValue = getFitMax();
    inRange = kFALSE ;
  }
  // test this value against our lower fit limit
  if(hasFitMin() && value < getFitMin()) {
    if(getFitMin() - value > 1e-6*range) {
      if (clippedValPtr)
	cout << "RooAbsRealLValue::inFitRange(" << GetName() << "): value " << value
	     << " rounded up to min limit " << getFitMin() << endl;
    }
    clippedValue = getFitMin();
    inRange = kFALSE ;
  } 

  if (clippedValPtr) *clippedValPtr=clippedValue ;
  return inRange ;
}



Bool_t RooAbsRealLValue::isValid(Double_t value, Bool_t verbose) const {
  if (!inFitRange(value)) {
    if (verbose)
      cout << "RooRealVar::isValid(" << GetName() << "): value " << value
           << " out of range" << endl ;
    return kFALSE ;
  }
  return kTRUE ;
}                                                                                                                         


Bool_t RooAbsRealLValue::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
}

void RooAbsRealLValue::writeToStream(ostream& os, Bool_t compact) const
{
}


Double_t RooAbsRealLValue::operator=(Double_t newValue) 
{
  Double_t clipValue ;
  // Clip 
  inFitRange(newValue,&clipValue) ;
  setVal(clipValue) ;

  return getVal() ;
}



void RooAbsRealLValue::copyCache(const RooAbsArg* source) 
{
  RooAbsReal::copyCache(source) ;
  setVal(_value) ; // force back-propagation
}



void RooAbsRealLValue::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this object to the specified stream. In addition to the info
  // from RooAbsReal::printToStream() we add:
  //
  //   Verbose : fit range and error

  RooAbsReal::printToStream(os,opt,indent);
  if(opt >= Verbose) {
    os << indent << "--- RooAbsRealLValue ---" << endl;
    TString unit(_unit);
    if(!unit.IsNull()) unit.Prepend(' ');
    os << indent << "  Fit range is [ ";
    if(hasFitMin()) {
      os << getFitMin() << unit << " , ";
    }
    else {
      os << "-INF , ";
    }
    if(hasFitMax()) {
      os << getFitMax() << unit << " ]" << endl;
    }
    else {
      os << "+INF ]" << endl;
    }
  }
}

