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
// BEGIN_HTML
// RooErrorVar is an auxilary class that represents the error
// of a RooRealVar as a seperate object. The main reason of
// existence of this class is to facilitate the reuse of existing
// techniques to perform calculations that involve a RooRealVars
// error, such as calculating the pull value.
// END_HTML
//
//

#include "RooFit.h"
#include "Riostream.h"

#include "RooErrorVar.h"
#include "RooErrorVar.h"
#include "RooAbsBinning.h"
#include "RooStreamParser.h"
#include "RooRangeBinning.h"
#include "RooMsgService.h"



ClassImp(RooErrorVar)
;



//_____________________________________________________________________________
RooErrorVar::RooErrorVar(const char *name, const char *title, const RooRealVar& input) :
  RooAbsRealLValue(name,title),
  _realVar("realVar","RooRealVar with error",this,(RooAbsReal&)input)
{
  // Construct an lvalue variable representing the error of RooRealVar input

  _binning = new RooUniformBinning(-1,1,100) ;
}



//_____________________________________________________________________________
RooErrorVar::RooErrorVar(const RooErrorVar& other, const char* name) :
  RooAbsRealLValue(other,name),
  _realVar("realVar",this,other._realVar)
{
  _binning = other._binning->clone() ;

  // Copy constructor

  TIterator* iter = other._altBinning.MakeIterator() ;
  RooAbsBinning* binning ;
  while((binning=(RooAbsBinning*)iter->Next())) {
    _altBinning.Add(binning->clone()) ;
  }
  delete iter ;
}



//_____________________________________________________________________________
RooErrorVar::~RooErrorVar()
{
  // Destructor 

  delete _binning ;
}



//_____________________________________________________________________________
Double_t RooErrorVar::getVal(const RooArgSet*) const 
{ 
  // Return value, i.e. error on input variable

  return evaluate();
}



//_____________________________________________________________________________
Bool_t RooErrorVar::hasBinning(const char* name) const
{
  // Return true if we have binning with given name
  
  return _altBinning.FindObject(name) ? kTRUE : kFALSE ;
}



//_____________________________________________________________________________
const RooAbsBinning& RooErrorVar::getBinning(const char* name, Bool_t verbose, Bool_t createOnTheFly) const 
{
  // Return binning with given name. If no binning exists with such a name, clone the default
  // binning on the fly if so requested

  return const_cast<RooErrorVar*>(this)->getBinning(name,verbose,createOnTheFly) ;
}



//_____________________________________________________________________________
RooAbsBinning& RooErrorVar::getBinning(const char* name, Bool_t /*verbose*/, Bool_t createOnTheFly) 
{
  // Return binning with given name. If no binning exists with such a name, clone the default
  // binning on the fly if so requested
  
  // Return default (normalization) binning and range if no name is specified
  if (name==0) {
    return *_binning ;
  }
  
  // Check if binning with this name has been created already
  RooAbsBinning* binning = (RooAbsBinning*) _altBinning.FindObject(name) ;
  if (binning) {
    return *binning ;
  }

  // Return default binning if binning is not found and no creation is requested
  if (!createOnTheFly) {
    return *_binning ;
  }

  // Create a new RooRangeBinning with this name with default range
  binning = new RooRangeBinning(getMin(),getMax(),name) ;
  coutI(Contents) << "RooErrorVar::getBinning(" << GetName() << ") new range named '" 
		  << name << "' created with default bounds" << endl ;

  _altBinning.Add(binning) ;

  return *binning ;
}



//_____________________________________________________________________________
void RooErrorVar::setBinning(const RooAbsBinning& binning, const char* name) 
{
  // Store given binning with this variable under the given name

  if (!name) {
    if (_binning) delete _binning ;
    _binning = binning.clone() ;
  } else {

    // Remove any old binning with this name
    RooAbsBinning* oldBinning = (RooAbsBinning*) _altBinning.FindObject(name) ;
    if (oldBinning) {
      _altBinning.Remove(oldBinning) ;
      delete oldBinning ;
    }

    // Insert new binning in list of alternative binnings
    RooAbsBinning* newBinning = binning.clone() ;
    newBinning->SetName(name) ;
    newBinning->SetTitle(name) ;
    _altBinning.Add(newBinning) ;

  }
  

}



//_____________________________________________________________________________
void RooErrorVar::setMin(const char* name, Double_t value) 
{
  // Set the lower bound of the range with the given name to the given value
  // If name is a null pointer, set the lower bound of the default range

  // Set new minimum of fit range 
  RooAbsBinning& binning = getBinning(name) ;

  // Check if new limit is consistent
  if (value >= getMax()) {
    coutW(InputArguments) << "RooErrorVar::setMin(" << GetName() 
			  << "): Proposed new fit min. larger than max., setting min. to max." << endl ;
    binning.setMin(getMax()) ;
  } else {
    binning.setMin(value) ;
  }

  // Clip current value in window if it fell out
  if (!name) {
    Double_t clipValue ;
    if (!inRange(_value,0,&clipValue)) {
      setVal(clipValue) ;
    }
  }
    
  setShapeDirty() ;
}


//_____________________________________________________________________________
void RooErrorVar::setMax(const char* name, Double_t value)
{
  // Set the upper bound of the range with the given name to the given value
  // If name is a null pointer, set the upper bound of the default range

  // Set new maximum of fit range 
  RooAbsBinning& binning = getBinning(name) ;

  // Check if new limit is consistent
  if (value < getMin()) {
    coutW(InputArguments) << "RooErrorVar::setMax(" << GetName() 
			  << "): Proposed new fit max. smaller than min., setting max. to min." << endl ;
    binning.setMax(getMin()) ;
  } else {
    binning.setMax(value) ;
  }

  // Clip current value in window if it fell out
  if (!name) {
    Double_t clipValue ;
    if (!inRange(_value,0,&clipValue)) {
      setVal(clipValue) ;
    }
  }

  setShapeDirty() ;
}



//_____________________________________________________________________________
void RooErrorVar::setRange( const char* name, Double_t min, Double_t max) 
{
  // Set the upper and lower lower bound of the range with the given name to the given values
  // If name is a null pointer, set the upper and lower bounds of the default range

  Bool_t exists = name ? (_altBinning.FindObject(name)?kTRUE:kFALSE) : kTRUE ;

  // Set new fit range 
  RooAbsBinning& binning = getBinning(name,kFALSE) ;

  // Check if new limit is consistent
  if (min>max) {
    coutW(InputArguments) << "RooErrorVar::setRange(" << GetName() 
			  << "): Proposed new fit max. smaller than min., setting max. to min." << endl ;
    binning.setRange(min,min) ;
  } else {
    binning.setRange(min,max) ;
  }

  if (!exists) {
    coutI(InputArguments) << "RooErrorVar::setRange(" << GetName() 
			  << ") new range named '" << name << "' created with bounds [" 
			  << min << "," << max << "]" << endl ;
  }

  setShapeDirty() ;  
}



//_____________________________________________________________________________
Bool_t RooErrorVar::readFromStream(istream& is, Bool_t /*compact*/, Bool_t verbose) 
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



//_____________________________________________________________________________
void RooErrorVar::writeToStream(ostream& os, Bool_t /*compact*/) const
{
  // Write value to stream

  os << getVal() ;
}


//_____________________________________________________________________________
void RooErrorVar::syncCache(const RooArgSet*) 
{ 
  // Force the internal value cache to be up to date

  _value = evaluate() ; 
}



