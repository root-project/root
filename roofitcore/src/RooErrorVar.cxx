/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooErrorVar.cc,v 1.3 2002/03/07 06:22:21 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   09-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [REAL] --
// RooErrorVar is an auxilary class that represents the error
// of a RooRealVar as a seperate object. The main reason of
// existence of this class is to facilitate the reuse of existing
// techniques to perform calculations that involve a RooRealVars
// error, such as calculating the pull value.
//

#include "RooFitCore/RooErrorVar.hh"
#include "RooFitCore/RooAbsBinning.hh"
#include "RooFitCore/RooStreamParser.hh"

ClassImp(RooErrorVar)
;


RooErrorVar::RooErrorVar(const char *name, const char *title, const RooRealVar& input) :
  RooAbsRealLValue(name,title),
  _realVar("realVar","RooRealVar with error",this,(RooAbsReal&)input)
{
  _binning = new RooUniformBinning(-1,1,100) ;
  // Constuctor
}


RooErrorVar::RooErrorVar(const RooErrorVar& other, const char* name) :
  RooAbsRealLValue(other,name),
  _realVar("realVar",this,other._realVar)
{
  _binning = other._binning->clone() ;

  // Copy constructor
}



RooErrorVar::~RooErrorVar()
{
  // Destructor 
  delete _binning ;
}



void RooErrorVar::setBinning(const RooAbsBinning& binning) 
{
  if (_binning) delete _binning ;
  _binning = binning.clone() ;
}



void RooErrorVar::setFitMin(Double_t value) 
{
  // Set new minimum of fit range 

  // Check if new limit is consistent
  if (value >= getFitMax()) {
    cout << "RooRealVar::setFitMin(" << GetName() 
	 << "): Proposed new fit min. larger than max., setting min. to max." << endl ;
    _binning->setMin(getFitMin()) ;
  } else {
    _binning->setMin(value) ;
  }

  // Clip current value in window if it fell out
  Double_t clipValue ;
  if (!inFitRange(_value,&clipValue)) {
    setVal(clipValue) ;
  }

  setShapeDirty() ;
}

void RooErrorVar::setFitMax(Double_t value)
{
  // Set new maximum of fit range 

  // Check if new limit is consistent
  if (value < getFitMin()) {
    cout << "RooRealVar::setFitMax(" << GetName() 
	 << "): Proposed new fit max. smaller than min., setting max. to min." << endl ;
    _binning->setMax(getFitMin()) ;
  } else {
    _binning->setMax(value) ;
  }

  // Clip current value in window if it fell out
  Double_t clipValue ;
  if (!inFitRange(_value,&clipValue)) {
    setVal(clipValue) ;
  }

  setShapeDirty() ;
}


void RooErrorVar::setFitRange(Double_t min, Double_t max) 
{
  // Set new fit range 

  // Check if new limit is consistent
  if (min>max) {
    cout << "RooRealVar::setFitRange(" << GetName() 
	 << "): Proposed new fit max. smaller than min., setting max. to min." << endl ;
    _binning->setRange(min,min) ;
  } else {
    _binning->setRange(min,max) ;
  }

  setShapeDirty() ;  
}



Bool_t RooErrorVar::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  // Read object contents from given stream

  TString token,errorPrefix("RooErrorVar::readFromStream(") ;
  errorPrefix.Append(GetName()) ;
  errorPrefix.Append(")") ;
  RooStreamParser parser(is,errorPrefix) ;
  Double_t value(0) ;

    // Compact mode: Read single token
  if (parser.readDouble(value,verbose)) return kTRUE ;
  if (isValidReal(value,verbose)) {
    setVal(value) ;
    return kFALSE ;
  } else {
    return kTRUE ;
  }
}


void RooErrorVar::writeToStream(ostream& os, Bool_t compact) const
{
  // Write value only
  os << getVal() ;
}
