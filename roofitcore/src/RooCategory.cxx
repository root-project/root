/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include <iostream.h>
#include <stdlib.h>
#include <string.h>
#include "TTree.h"
#include "TString.h"
#include "TH1.h"
#include "RooFitCore/RooCategory.hh"
#include "RooFitCore/RooArgSet.hh"

ClassImp(RooCategory) 
;


RooCategory::RooCategory(const char *name, const char *title) : 
  RooAbsCategory(name,title)
{
  setValueDirty(kTRUE) ;  
  setShapeDirty(kTRUE) ;  
}


RooCategory::RooCategory(const RooCategory& other) :
  RooAbsCategory(other)
{
  setValueDirty(kTRUE) ;  
  setShapeDirty(kTRUE) ;  
}


RooCategory::~RooCategory()
{
}


RooAbsArg& RooCategory::operator=(RooAbsArg& aother)
{
  RooAbsArg::operator=(aother) ;
  RooCategory& other=(RooCategory&)aother ;

  setValueDirty(kTRUE) ;
  setShapeDirty(kTRUE) ;
}



Bool_t RooCategory::setIndex(Int_t index) 
{
  RooCatType* type ;  
  for (int i=0 ; i<_types.GetEntries() ; i++) {
    RooCatType& entry = *(RooCatType*)_types.At(i) ;
    if (entry == index) {      
      _value = entry ;
      setValueDirty(kTRUE) ;
      return kFALSE ;
    }
  }
  cout << "Rooindex::setIndex(" << GetName() << "): index " << index << " is not defined" << endl ;
  return kTRUE ;  
}



Bool_t RooCategory::setLabel(char* label) 
{
  RooCatType* type ;  
  for (int i=0 ; i<_types.GetEntries() ; i++) {
    RooCatType& entry = *(RooCatType*)_types.At(i) ;
    if (entry==label) {
      _value = entry ;
      setValueDirty(kTRUE) ;
      return kFALSE ;
    }
  }
  cout << "Rooindex::setIndex(" << GetName() << "): label " << label << " is not defined" << endl ;
  return kTRUE ;  
}



Bool_t RooCategory::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  // Read object contents from given stream

  // compact only at the moment
  // Read single token
  TString token ;
  is >> token ;

  // Convert token to double
  char *endptr(0) ;
  Int_t index = strtol(token.Data(),&endptr,10) ;	  
  int nscan = endptr-((const char *)token.Data()) ;	  
  if (nscan<token.Length() && !token.IsNull()) {
    if (verbose) {
      cout << "RooCategory::readFromStream(" << GetName() 
	   << "): cannot convert token \"" << token 
	   << "\" to integer number" << endl ;
    }
    return kTRUE ;
  }

  if (isValidIndex(index)) {
    setIndex(index) ;
    return kFALSE ;  
  } else {
    if (verbose) {
      cout << "RooCategory::readFromStream(" << GetName() 
	   << "): index undefined: " << index << endl ;
    }
    return kTRUE;
  }
}



void RooCategory::writeToStream(ostream& os, Bool_t compact)
{
  // Write object contents to given stream

  // compact only at the moment
  os << _value.getVal() ;
}



void RooCategory::attachToTree(TTree& t, Int_t bufSize)
{
  // Attach object to a branch of given TTree

  // First determine if branch is taken
  if (t.GetBranch(GetName())) {
    //cout << "RooRealVar::attachToTree(" << GetName() << "): branch in tree " << t.GetName() << " already exists" << endl ;
    t.SetBranchAddress(GetName(),&((Int_t&)_value)) ;
  } else {    
    TString format(GetName());
    format.Append("/I");
    t.Branch(GetName(), &((Int_t&)_value), (const Text_t*)format, bufSize);
  }
}


void RooCategory::printToStream(ostream& os, PrintOption opt) 
{
  if (_types.GetEntries()==0) {
    os << "RooCategory: " << GetName() << " has no types defined" << endl ;
    return ;
  }

  //Print object contents
  if (opt==Shape) {
    RooAbsCategory::printToStream(os,Shape) ;
  } else {
    os << "RooCategory: " << GetName() << " = " << getLabel() << "(" << getIndex() << ")" ;
    os << " : \"" << fTitle << "\"" ;
    
    printAttribList(os) ;
    os << endl ;
  }
}



