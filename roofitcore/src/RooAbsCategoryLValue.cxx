/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsCategoryLValue.cc,v 1.23 2005/06/20 15:44:44 wverkerke Exp $
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

// -- CLASS DESCRIPTION [CAT] --
// RooAbsCategoryLValue is the common abstract base class for objects that represent a
// discrete value that may appear on the left hand side of an equation ('lvalue')
//
// Each implementation must provide setIndex()/setLabel() members to allow direct modification 
// of the value. RooAbsCategoryLValue may be derived, but its functional relation
// to other RooAbsArgs must be invertible
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

ClassImp(RooAbsCategoryLValue) 
;


RooAbsCategoryLValue::RooAbsCategoryLValue(const char *name, const char *title) : 
  RooAbsCategory(name,title)
{
  // Constructor
  setValueDirty() ;  
  setShapeDirty() ;  
}


RooAbsCategoryLValue::RooAbsCategoryLValue(const RooAbsCategoryLValue& other, const char* name) :
  RooAbsCategory(other, name), RooAbsLValue(other)
{
  // Copy constructor
}


RooAbsCategoryLValue::~RooAbsCategoryLValue()
{
  // Destructor
}


RooAbsArg& RooAbsCategoryLValue::operator=(Int_t index) 
{
  // Assignment operator from integer index number
  setIndex(index,kTRUE) ;
  return *this ;
}


RooAbsArg& RooAbsCategoryLValue::operator=(const char *label) 
{
  // Assignment operator from string pointer
  setLabel(label) ;
  return *this ;
}

RooAbsArg& RooAbsCategoryLValue::operator=(const RooAbsCategory& other) 
{
  // Assignment from another RooCategory
  if (&other==this) return *this ;

  const RooCatType* type = lookupType(other.getLabel(),kTRUE) ;
  if (!type) return *this ;

  _value = *type ;
  setValueDirty() ;
  return *this ;
}


Bool_t RooAbsCategoryLValue::setOrdinal(UInt_t n) 
{
  // Set our state to our n'th defined type and return kTRUE.
  // Return kFALSE if n is out of range.

  const RooCatType *newValue= getOrdinal(n);
  if(newValue) {
    return setIndex(newValue->getVal());
  }
  else {
    return kFALSE;
  }
}

void RooAbsCategoryLValue::copyCache(const RooAbsArg* source) 
{
  // copy cached value from another object
  RooAbsCategory::copyCache(source) ;
  setIndex(_value.getVal()) ; // force back-propagation
}



Bool_t RooAbsCategoryLValue::readFromStream(istream&, Bool_t, Bool_t) 
{
  // Read object contents from given stream
  return kTRUE ;
}



void RooAbsCategoryLValue::writeToStream(ostream&, Bool_t) const
{
  // Write object contents to given stream
}



void RooAbsCategoryLValue::randomize() {
  // Randomize current value
  UInt_t ordinal= RooRandom::integer(numTypes());
  setOrdinal(ordinal);
}



void RooAbsCategoryLValue::setBin(Int_t ibin) 
{
  // Set category to i-th fit bin, which is the i-th registered state.

  // Check validity of ibin
  if (ibin<0 || ibin>=numBins()) {
    cout << "RooAbsCategoryLValue::setBin(" << GetName() << ") ERROR: bin index " << ibin
	 << " is out of range (0," << numBins()-1 << ")" << endl ;
    return ;
  }

  // Retrieve state corresponding to bin
  const RooCatType* type = getOrdinal(ibin) ;

  // Set value to requested state
  setIndex(type->getVal()) ;
}



Int_t RooAbsCategoryLValue::getBin() const 
{
  // Get index of plot bin for current value this category.

  //Synchronize _value
  getIndex() ; 

  // Lookup ordinal index number 
  return _types.IndexOf(_types.FindObject(_value.GetName())) ;
}



Int_t RooAbsCategoryLValue::numBins() const 
{
  // Returm the number of fit bins ( = number of types )
  return numTypes() ;
}
