/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UCSB, verkerke@slac.stanford.edu
 * History:
 *   01-Mar-2001 WV Create initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [CAT] --
// RooMultiCategory consolidates several RooAbsCategory objects into
// a single category. The states of the multi-category consist of all the permutations
// of the input categories. 
//
// RooMultiCategory state are automatically defined and updated whenever an input
// category modifies its list of states

#include <iostream.h>
#include <stdlib.h>
#include <stdio.h>
#include "TString.h"
#include "RooFitCore/RooMultiCategory.hh"
#include "RooFitCore/RooStreamParser.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooMultiCatIter.hh"
#include "RooFitCore/RooAbsCategory.hh"

ClassImp(RooMultiCategory)
;

RooMultiCategory::RooMultiCategory(const char *name, const char *title, const RooArgSet& inputCatList) :
  RooAbsCategory(name, title), _catSet("catSet","Input category set",this,kTRUE,kTRUE)
{  
  // Constructor from list of input categories

  // Copy category list
  TIterator* iter = inputCatList.createIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)iter->Next()) {
    if (!dynamic_cast<RooAbsCategory*>(arg)) {
      cout << "RooMultiCategory::RooMultiCategory(" << GetName() << "): input argument " << arg->GetName() 
	   << " is not a RooAbsCategory" << endl ;
    }
    _catSet.add(*arg) ;
  }
  delete iter ;
  
  updateIndexList() ;
}


RooMultiCategory::RooMultiCategory(const RooMultiCategory& other, const char *name) :
  RooAbsCategory(other,name), _catSet("catSet",this,other._catSet)
{
  // Copy constructor
  updateIndexList() ;
}



RooMultiCategory::~RooMultiCategory() 
{
  // Destructor
}



void RooMultiCategory::updateIndexList()
{
  // Update the list of super-category states 

  clearTypes() ;

  RooMultiCatIter iter(_catSet) ;
  TObjString* obj ;
  while(obj=(TObjString*)iter.Next()) {
    // Register composite label
    defineType(obj->String()) ;
  }

  // Renumbering will invalidate cache
  setValueDirty() ;
}


TString RooMultiCategory::currentLabel() const
{
  // Return the name of the current state, 
  // constructed from the state names of the input categories

  TIterator* lIter = _catSet.createIterator() ;

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
RooMultiCategory::evaluate() const
{
  // Calculate the current value 
  if (isShapeDirty()) const_cast<RooMultiCategory*>(this)->updateIndexList() ;
  return *lookupType(currentLabel()) ;
}



void RooMultiCategory::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  // Print the state of this object to the specified output stream.

  RooAbsCategory::printToStream(os,opt,indent) ;
  
  if (opt>=Verbose) {     
    os << indent << "--- RooMultiCategory ---" << endl;
    os << indent << "  Input category list:" << endl ;
    TString moreIndent(indent) ;
    moreIndent.Append("   ") ;
    _catSet.printToStream(os,Standard,moreIndent.Data()) ;
  }
}


Bool_t RooMultiCategory::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  // Read object contents from given stream
  return kTRUE ;
}



void RooMultiCategory::writeToStream(ostream& os, Bool_t compact) const
{
  // Write object contents to given stream
  RooAbsCategory::writeToStream(os,compact) ;
}
