/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooSuperCategory.cc,v 1.8 2001/07/31 05:54:22 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UCSB, verkerke@slac.stanford.edu
 * History:
 *   01-Mar-2001 WV Create initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooSuperCategory consolidates several RooAbsCategoryLValue objects into
// a single category. The states of the super category consist of all the permutations
// of the input categories. The super category is an lvalue itself and a modification
// of its state will back propagate into a modification of its input categories.
//
// RooSuperCategory state are automatically defined and updated whenever an input
// category modifies its list of states

#include <iostream.h>
#include <stdlib.h>
#include <stdio.h>
#include "TString.h"
#include "RooFitCore/RooSuperCategory.hh"
#include "RooFitCore/RooStreamParser.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooMultiCatIter.hh"
#include "RooFitCore/RooAbsCategoryLValue.hh"

ClassImp(RooSuperCategory)
;

RooSuperCategory::RooSuperCategory(const char *name, const char *title, const RooArgSet& inputCatList) :
  RooAbsCategoryLValue(name, title), _catSet("catSet","Input category set",this,kTRUE,kTRUE)
{  
  // Constructor from list of input categories

  // Copy category list
  TIterator* iter = inputCatList.MakeIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)iter->Next()) {
    if (!arg->IsA()->InheritsFrom(RooAbsCategoryLValue::Class())) {
      cout << "RooSuperCategory::RooSuperCategory(" << GetName() << "): input category " << arg->GetName() 
	   << " is not an lvalue" << endl ;
    }
    _catSet.add(*arg) ;
  }
  delete iter ;
  
  updateIndexList() ;
}


RooSuperCategory::RooSuperCategory(const RooSuperCategory& other, const char *name) :
  RooAbsCategoryLValue(other,name), _catSet("catSet",this,other._catSet)
{
  updateIndexList() ;
  setIndex(other.getIndex()) ;
}



RooSuperCategory::~RooSuperCategory() 
{
  // Destructor
}



TIterator* RooSuperCategory::MakeIterator() const 
{
  // Make an iterator over the input categories of this supercategory
  return new RooMultiCatIter(_catSet) ;
}



void RooSuperCategory::updateIndexList()
{
  // Update the list of our category states 

  clearTypes() ;
  RooArgSet* catListClone = _catSet.snapshot(kTRUE) ;
  RooMultiCatIter mcIter(_catSet) ;

  while(mcIter.Next()) {
    // Register composite label
    defineType(currentLabel()) ;
  }

  _catSet = *catListClone ;
  delete catListClone ;

  // Renumbering will invalidate cache
  setValueDirty() ;
}


TString RooSuperCategory::currentLabel() const
{
  // Return the name of the current state, 
  // constructed from the state names of the input categories

  TIterator* lIter = _catSet.MakeIterator() ;

  // Construct composite label name
  TString label ;
  RooAbsCategory* cat ;
  Bool_t first(kTRUE) ;
  while(cat=(RooAbsCategory*) lIter->Next()) {
    label.Append(first?"{":";") ;
    label.Append(cat->getLabel()) ;      
    first=kFALSE ;
  }
  label.Append("}") ;  
  delete lIter ;

  return label ;
}


RooCatType
RooSuperCategory::evaluate() const
{
  // Calculate the current value 
  if (isShapeDirty()) const_cast<RooSuperCategory*>(this)->updateIndexList() ;
  return *lookupType(currentLabel()) ;
}


Bool_t RooSuperCategory::setIndex(Int_t index, Bool_t printError) 
{
  // Set the value of the super category by specifying the state index code
  // Indirectly sets the values of the input categories
  const RooCatType* type = lookupType(index,kTRUE) ;
  if (!type) return kTRUE ;
  return setType(type) ;
}


Bool_t RooSuperCategory::setLabel(const char* label, Bool_t printError) 
{
  // Set the value of the super category by specifying the state name
  // Indirectly sets the values of the input categories
  const RooCatType* type = lookupType(label,kTRUE) ;
  if (!type) return kTRUE ;
  return setType(type) ;
}


Bool_t RooSuperCategory::setType(const RooCatType* type, Bool_t printError)
{
  // Set the value of the super category by specifying the state object
  // Indirectly sets the values of the input categories

  // WVE: adapt parser to understand composite super categories
  char buf[1024] ;
  strcpy(buf,type->GetName()) ;

  TIterator* iter = _catSet.MakeIterator() ;
  RooAbsCategoryLValue* arg ;
  Bool_t error(kFALSE) ;

  // Parse composite label and set label of components to their values
  char* ptr = strtok(buf+1,";}") ;
  while (arg=(RooAbsCategoryLValue*)iter->Next()) {
    error |= arg->setLabel(ptr) ;
    ptr = strtok(0,";}") ;
  }
  
  delete iter ;
  return error ;
}



void RooSuperCategory::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  // Print the state of this object to the specified output stream.

  RooAbsCategory::printToStream(os,opt,indent) ;
  
  if (opt>=Verbose) {     
    os << indent << "--- RooSuperCategory ---" << endl;
    os << indent << "  Input category list:" << endl ;
    _catSet.printToStream(os,Standard,TString(indent).Append("  ")) ;
  }
}


Bool_t RooSuperCategory::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  // Read object contents from given stream
  return kTRUE ;
}



void RooSuperCategory::writeToStream(ostream& os, Bool_t compact) const
{
  // Write object contents to given stream
  RooAbsCategory::writeToStream(os,compact) ;
}
