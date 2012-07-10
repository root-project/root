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
// RooAbsCategoryLValue is the common abstract base class for objects that represent a
// discrete value that may appear on the left hand side of an equation ('lvalue')
//
// Each implementation must provide setIndex()/setLabel() members to allow direct modification 
// of the value. RooAbsCategoryLValue may be derived, but its functional relation
// to other RooAbsArgs must be invertible
// END_HTML
//
//

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <stdlib.h>
#include <string.h>
#include "TTree.h"
#include "TString.h"
#include "TH1.h"
#include "RooAbsCategoryLValue.h"
#include "RooArgSet.h"
#include "RooStreamParser.h"
#include "RooRandom.h"
#include "RooMsgService.h"

using namespace std;

ClassImp(RooAbsCategoryLValue) 
;


//_____________________________________________________________________________
RooAbsCategoryLValue::RooAbsCategoryLValue(const char *name, const char *title) : 
  RooAbsCategory(name,title)
{
  // Constructor

  setValueDirty() ;  
  setShapeDirty() ;  
}



//_____________________________________________________________________________
RooAbsCategoryLValue::RooAbsCategoryLValue(const RooAbsCategoryLValue& other, const char* name) :
  RooAbsCategory(other, name), RooAbsLValue(other)
{
  // Copy constructor
}



//_____________________________________________________________________________
RooAbsCategoryLValue::~RooAbsCategoryLValue()
{
  // Destructor
}



//_____________________________________________________________________________
RooAbsArg& RooAbsCategoryLValue::operator=(Int_t index) 
{
  // Assignment operator from integer index number

  setIndex(index,kTRUE) ;
  return *this ;
}



//_____________________________________________________________________________
RooAbsArg& RooAbsCategoryLValue::operator=(const char *label) 
{
  // Assignment operator from string pointer

  setLabel(label) ;
  return *this ;
}



//_____________________________________________________________________________
RooAbsArg& RooAbsCategoryLValue::operator=(const RooAbsCategory& other) 
{
  // Assignment from another RooAbsCategory

  if (&other==this) return *this ;

  const RooCatType* type = lookupType(other.getLabel(),kTRUE) ;
  if (!type) return *this ;

  _value = *type ;
  setValueDirty() ;
  return *this ;
}



//_____________________________________________________________________________
Bool_t RooAbsCategoryLValue::setOrdinal(UInt_t n, const char* rangeName) 
{
  // Set our state to our n'th defined type and return kTRUE.
  // Return kFALSE if n is out of range.

  const RooCatType *newValue= getOrdinal(n,rangeName);
  if(newValue) {
    return setIndex(newValue->getVal());
  }
  else {
    return kFALSE;
  }
}



//_____________________________________________________________________________
void RooAbsCategoryLValue::copyCache(const RooAbsArg* source, Bool_t valueOnly, Bool_t setValDirty) 
{
  // Copy the cached value from given source and raise dirty flag.
  // It is the callers responsability to ensure that the sources
  // cache is clean(valid) before this function is called, e.g. by
  // calling syncCache() on the source.

  RooAbsCategory::copyCache(source,valueOnly,setValDirty) ;
  if (isValid(_value)) {
    setIndex(_value.getVal()) ; // force back-propagation
  }
}



//_____________________________________________________________________________
Bool_t RooAbsCategoryLValue::readFromStream(istream&, Bool_t, Bool_t) 
{
  // Read object contents from given stream (dummy implementation)

  return kTRUE ;
}



//_____________________________________________________________________________
void RooAbsCategoryLValue::writeToStream(ostream&, Bool_t) const
{
  // Write object contents to given stream (dummy implementation)
}



//_____________________________________________________________________________
void RooAbsCategoryLValue::randomize(const char* rangeName) 
{
  // Randomize current value
  
  UInt_t ordinal= RooRandom::integer(numTypes(rangeName));
  setOrdinal(ordinal,rangeName);
}



//_____________________________________________________________________________
void RooAbsCategoryLValue::setBin(Int_t ibin, const char* rangeName) 
{
  // Set category to i-th fit bin, which is the i-th registered state.

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



//_____________________________________________________________________________
Int_t RooAbsCategoryLValue::getBin(const char* /*rangeName*/) const 
{
  // Get index of plot bin for current value this category.

  //Synchronize _value
  getLabel() ; 
  
  // Lookup ordinal index number 
  return _types.IndexOf(_types.FindObject(_value.GetName())) ;
}



//_____________________________________________________________________________
Int_t RooAbsCategoryLValue::numBins(const char* rangeName) const 
{
  // Returm the number of fit bins ( = number of types )

  return numTypes(rangeName) ;
}
