/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCategory.cc,v 1.14 2001/05/17 00:43:15 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooCategory represents a fundamental (non-derived) discrete value object. The class
// has a public interface to define the possible value states.


#include <iostream.h>
#include <stdlib.h>
#include <string.h>
#include "TTree.h"
#include "TString.h"
#include "TH1.h"
#include "RooFitCore/RooCategory.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooStreamParser.hh"

ClassImp(RooCategory) 
;


RooCategory::RooCategory(const char *name, const char *title) : 
  RooAbsCategoryLValue(name,title)
{
  // Constructor. Types must be defined before variable can be used
  setValueDirty() ;  
  setShapeDirty() ;  
}


RooCategory::RooCategory(const RooCategory& other, const char* name) :
  RooAbsCategoryLValue(other, name)
{
  // Copy constructor
}


RooCategory::~RooCategory()
{
  // Destructor
}



Bool_t RooCategory::setIndex(Int_t index, Bool_t printError) 
{
  // Set value by specifying the index code of the desired state

  const RooCatType* type = lookupType(index,printError) ;
  if (!type) return kTRUE ;
  _value = *type ;
  setValueDirty() ;
  return kFALSE ;
}



Bool_t RooCategory::setLabel(const char* label, Bool_t printError) 
{
  // Set value by specifying the name of the desired state

  const RooCatType* type = lookupType(label,printError) ;
  if (!type) return kTRUE ;
  _value = *type ;
  setValueDirty() ;
  return kFALSE ;
}



Bool_t RooCategory::defineType(const char* label) 
{ 
  // Define a state with given name, index code as automatically assigned

  if (TString(label).Contains(";")) {
  cout << "RooCategory::defineType(" << GetName() 
       << "): semicolons not allowed in label name" << endl ;
  return kTRUE ;
  }

  return RooAbsCategory::defineType(label)?kFALSE:kTRUE ; 
}


Bool_t RooCategory::defineType(const char* label, Int_t index) 
{
  // Define a state with given name and index code

  if (TString(label).Contains(";")) {
  cout << "RooCategory::defineType(" << GetName() 
       << "): semicolons not allowed in label name" << endl ;
  return kTRUE ;
  }

  return RooAbsCategory::defineType(label,index)?kFALSE:kTRUE ; 
}


RooCategory& RooCategory::operator=(const RooCategory& other) 
{
  // Assignment from another RooCategory

  const RooCatType* type = lookupType(other._value,kTRUE) ;
  if (!type) return *this ;

  _value = *type ;
  setValueDirty() ;
  return *this ;
}


Bool_t RooCategory::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  // Read object contents from given stream

  // Read single token
  RooStreamParser parser(is) ;
  TString token = parser.readToken() ;

  return setLabel(token,verbose) ;
}



void RooCategory::writeToStream(ostream& os, Bool_t compact) const
{
  // compact only at the moment
  if (compact) {
    os << getIndex() ;
  } else {
    os << getLabel() ;
  }
}




