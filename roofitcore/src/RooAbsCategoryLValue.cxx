/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsCategoryLValue.cc,v 1.5 2001/05/31 21:21:34 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooAbsCategoryLValue is the common abstract base class for objects that represent a
// discrete value that may appear on the left hand side of an equation ('lvalue')
//
// Each implementation must provide setIndex()/setLabel() members to allow direct modification 
// of the value. RooAbsCategoryLValue may be derived, but its functional relation
// to other RooAbsArgs must be invertible
//

#include <iostream.h>
#include <stdlib.h>
#include <string.h>
#include "TTree.h"
#include "TString.h"
#include "TH1.h"
#include "RooFitCore/RooAbsCategoryLValue.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooStreamParser.hh"
#include "RooFitCore/RooGenContext.hh"

ClassImp(RooAbsCategoryLValue) 
;


RooAbsCategoryLValue::RooAbsCategoryLValue(const char *name, const char *title) : 
  RooAbsCategory(name,title)
{
  // Constructor
  setValueDirty(kTRUE) ;  
  setShapeDirty(kTRUE) ;  
}


RooAbsCategoryLValue::RooAbsCategoryLValue(const RooAbsCategoryLValue& other, const char* name) :
  RooAbsCategory(other, name)
{
  // Copy constructor
}


RooAbsCategoryLValue::~RooAbsCategoryLValue()
{
  // Destructor
}


RooAbsCategoryLValue& RooAbsCategoryLValue::operator=(Int_t index) 
{
  // Assignment operator from integer index number
  setIndex(index,kTRUE) ;
  return *this ;
}


RooAbsCategoryLValue& RooAbsCategoryLValue::operator=(const char *label) 
{
  // Assignment operator from string pointer
  setLabel(label) ;
  return *this ;
}

Bool_t RooAbsCategoryLValue::setOrdinal(UInt_t n) {
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



Bool_t RooAbsCategoryLValue::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  // Read object contents from given stream
  return kTRUE ;
}



void RooAbsCategoryLValue::writeToStream(ostream& os, Bool_t compact) const
{
  // Write object contents to given stream
}

void RooAbsCategoryLValue::randomize() {
  UInt_t ordinal= RooGenContext::integer(numTypes());
  setOrdinal(ordinal);
}
