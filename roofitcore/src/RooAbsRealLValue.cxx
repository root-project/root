/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsRealLValue.cc,v 1.11 2001/08/22 00:50:24 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooAbsRealLValue is the common abstract base class for objects that represent a
// real value that may appear on the left hand side of an equation ('lvalue')
// Each implementation must provide a setVal() member to allow direct modification 
// of the value. RooAbsRealLValue may be derived, but its functional relation
// to other RooAbsArg must be invertible
//
// This class has methods that export a fit range, but doesn't hold its values
// because these limits may be derived from limits of client object.
// The fit limits serve as integration range when interpreted
// as a dependent and a boundaries when interpreted as a parameter.

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "TObjString.h"
#include "TTree.h"
#include "RooFitCore/RooAbsRealLValue.hh"
#include "RooFitCore/RooStreamParser.hh"
#include "RooFitCore/RooRandom.hh"

ClassImp(RooAbsRealLValue)

RooAbsRealLValue::RooAbsRealLValue(const char *name, const char *title, const char *unit) :
  RooAbsReal(name, title, 0, 0, unit)
{
  // Constructor
}  


RooAbsRealLValue::RooAbsRealLValue(const RooAbsRealLValue& other, const char* name) :
  RooAbsReal(other,name)
{
  // Copy constructor
}


RooAbsRealLValue::~RooAbsRealLValue() 
{
  // Destructor
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



Bool_t RooAbsRealLValue::isValidReal(Double_t value, Bool_t verbose) const 
{
  // Check if given value is valid
  if (!inFitRange(value)) {
    if (verbose)
      cout << "RooRealVar::isValid(" << GetName() << "): value " << value
           << " out of range (" << getFitMin() << " - " << getFitMax() << ")" << endl ;
    return kFALSE ;
  }
  return kTRUE ;
}                                                                                                                         


Bool_t RooAbsRealLValue::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  // Read object contents from given stream
  return kTRUE ;
}

void RooAbsRealLValue::writeToStream(ostream& os, Bool_t compact) const
{
  // Write object contents to given stream
}


RooAbsRealLValue& RooAbsRealLValue::operator=(Double_t newValue) 
{
  // Assignment operator from a Double_t

  Double_t clipValue ;
  // Clip 
  inFitRange(newValue,&clipValue) ;
  setVal(clipValue) ;

  return *this ;
}


RooAbsRealLValue& RooAbsRealLValue::operator=(const RooAbsReal& arg) 
{
  return operator=(arg.getVal()) ;
}





void RooAbsRealLValue::copyCache(const RooAbsArg* source) 
{
  // Copy cache of another RooAbsArg to our cache

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

void RooAbsRealLValue::randomize() {
  // Set a new value sampled from a uniform distribution over the fit range.
  // Prints a warning and does nothing if the fit range is not finite.

  if(hasFitMin() && hasFitMax()) {
    Double_t range= getFitMax()-getFitMin();
    setVal(getFitMin() + RooRandom::uniform()*range);
  }
  else {
    cout << fName << "::" << ClassName() << ":randomize: fails with unbounded fit range" << endl;
  }
}


void RooAbsRealLValue::setPlotBin(Int_t ibin) 
{
  // Check range of plot bin index
  if (ibin<0 || ibin>=numPlotBins()) {
    cout << "RooAbsRealLValue::setPlotBin(" << GetName() << ") ERROR: bin index " << ibin
	 << " is out of range (0," << getPlotBins()-1 << ")" << endl ;
    return ;
  }
 
  // Set value to center of requested bin
  setVal(plotBinCenter(ibin)) ;
}
