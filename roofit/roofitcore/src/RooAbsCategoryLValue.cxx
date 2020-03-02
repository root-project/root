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

/**
\file RooAbsCategoryLValue.cxx
\class RooAbsCategoryLValue
\ingroup Roofitcore

RooAbsCategoryLValue is the common abstract base class for objects that represent a
discrete value that may appear on the left hand side of an equation ('lvalue')

Each implementation must provide setIndex()/setLabel() members to allow direct modification 
of the value. RooAbsCategoryLValue may be derived, but its functional relation
to other RooAbsArgs must be invertible
**/

#include "RooFit.h"

#include "Riostream.h"
#include <stdlib.h>
#include "TTree.h"
#include "TString.h"
#include "TH1.h"
#include "RooAbsCategoryLValue.h"
#include "RooArgSet.h"
#include "RooStreamParser.h"
#include "RooRandom.h"
#include "RooMsgService.h"

using namespace std;

ClassImp(RooAbsCategoryLValue); 



////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooAbsCategoryLValue::RooAbsCategoryLValue(const char *name, const char *title) : 
  RooAbsCategory(name,title)
{
  setValueDirty() ;  
  setShapeDirty() ;  
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooAbsCategoryLValue::RooAbsCategoryLValue(const RooAbsCategoryLValue& other, const char* name) :
  RooAbsCategory(other, name), RooAbsLValue(other)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooAbsCategoryLValue::~RooAbsCategoryLValue()
{
}



////////////////////////////////////////////////////////////////////////////////
/// Assignment operator from integer index number

RooAbsArg& RooAbsCategoryLValue::operator=(Int_t index) 
{
  setIndex(index,kTRUE) ;
  return *this ;
}



////////////////////////////////////////////////////////////////////////////////
/// Assignment operator from string pointer

RooAbsArg& RooAbsCategoryLValue::operator=(const char *label) 
{
  setLabel(label) ;
  return *this ;
}



////////////////////////////////////////////////////////////////////////////////
/// Assignment from another RooAbsCategory

RooAbsArg& RooAbsCategoryLValue::operator=(const RooAbsCategory& other) 
{
  if (&other==this) return *this ;

  const RooCatType* type = lookupType(other.getLabel(),kTRUE) ;
  if (!type) return *this ;

  _value = *type ;
  setValueDirty() ;
  return *this ;
}



////////////////////////////////////////////////////////////////////////////////
/// Set our state to our n'th defined type and return kTRUE.
/// Return kFALSE if n is out of range.

Bool_t RooAbsCategoryLValue::setOrdinal(UInt_t n, const char* rangeName) 
{
  const RooCatType *newValue= getOrdinal(n,rangeName);
  if(newValue) {
    return setIndex(newValue->getVal());
  }
  else {
    return kFALSE;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Copy the cached value from given source and raise dirty flag.
/// It is the callers responsability to ensure that the sources
/// cache is clean(valid) before this function is called, e.g. by
/// calling syncCache() on the source.

void RooAbsCategoryLValue::copyCache(const RooAbsArg* source, Bool_t valueOnly, Bool_t setValDirty) 
{
  RooAbsCategory::copyCache(source,valueOnly,setValDirty) ;
  if (isValid(_value)) {
    setIndex(_value.getVal()) ; // force back-propagation
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Read object contents from given stream (dummy implementation)

Bool_t RooAbsCategoryLValue::readFromStream(istream&, Bool_t, Bool_t) 
{
  return kTRUE ;
}



////////////////////////////////////////////////////////////////////////////////
/// Write object contents to given stream (dummy implementation)

void RooAbsCategoryLValue::writeToStream(ostream&, Bool_t) const
{
}



////////////////////////////////////////////////////////////////////////////////
/// Randomize current value

void RooAbsCategoryLValue::randomize(const char* rangeName) 
{
  UInt_t ordinal= RooRandom::integer(numTypes(rangeName));
  setOrdinal(ordinal,rangeName);
}



////////////////////////////////////////////////////////////////////////////////
/// Set category to i-th fit bin, which is the i-th registered state.

void RooAbsCategoryLValue::setBin(Int_t ibin, const char* rangeName) 
{
  // Check validity of ibin
  if (ibin<0 || ibin>=numBins(rangeName)) {
    coutE(InputArguments) << "RooAbsCategoryLValue::setBin(" << GetName() << ") ERROR: bin index " << ibin
			  << " is out of range (0," << numBins(rangeName)-1 << ")" << endl ;
    return ;
  }

  // Retrieve state corresponding to bin
  const RooCatType* type = getOrdinal(ibin,rangeName) ;

  // Set value to requested state
  setIndex(type->getVal()) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Get index of plot bin for current value this category.

Int_t RooAbsCategoryLValue::getBin(const char* /*rangeName*/) const 
{
  //Synchronize _value
  getLabel() ; 
  
  // Lookup ordinal index number
  std::string theName = _value.GetName();
  auto item = std::find_if(_types.begin(), _types.end(), [&theName](const RooCatType* cat){
    return cat->GetName() == theName;
  });

  return item - _types.begin();
}



////////////////////////////////////////////////////////////////////////////////
/// Returm the number of fit bins ( = number of types )

Int_t RooAbsCategoryLValue::numBins(const char* rangeName) const 
{
  return numTypes(rangeName) ;
}
