/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooErrorVar.cc,v 1.1 2001/10/11 01:28:50 verkerke Exp $
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

ClassImp(RooErrorVar)
;


RooErrorVar::RooErrorVar(const char *name, const char *title, const RooRealVar& input) :
  RooAbsRealLValue(name,title),
  _realVar("realVar","RooRealVar with error",this,(RooAbsReal&)input)
{
  // Constuctor
}


RooErrorVar::RooErrorVar(const RooErrorVar& other, const char* name) :
  RooAbsRealLValue(other,name),
  _realVar("realVar",this,other._realVar)
{
  // Copy constructor
}


void RooErrorVar::setFitMin(Double_t value) 
{
  // Set new minimum of fit range 

  // Check if new limit is consistent
  if (value >= _fitMax) {
    cout << "RooRealVar::setFitMin(" << GetName() 
	 << "): Proposed new fit min. larger than max., setting min. to max." << endl ;
    _fitMin = _fitMax ;
  } else {
    _fitMin = value ;
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
  if (value < _fitMin) {
    cout << "RooRealVar::setFitMax(" << GetName() 
	 << "): Proposed new fit max. smaller than min., setting max. to min." << endl ;
    _fitMax = _fitMin ;
  } else {
    _fitMax = value ;
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
    _fitMin = min ;
    _fitMax = min ;
  } else {
    _fitMin = min ;
    _fitMax = max ;
  }

  setShapeDirty() ;  
}
