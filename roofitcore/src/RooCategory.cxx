/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCategory.cc,v 1.5 2001/03/22 15:31:25 verkerke Exp $
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
#include "RooFitCore/RooStreamParser.hh"

ClassImp(RooCategory) 
;


RooCategory::RooCategory(const char *name, const char *title) : 
  RooAbsCategory(name,title)
{
  setValueDirty(kTRUE) ;  
  setShapeDirty(kTRUE) ;  
}


RooCategory::RooCategory(const char* name, const RooCategory& other) :
  RooAbsCategory(name, other)
{
}


RooCategory::RooCategory(const RooCategory& other) :
  RooAbsCategory(other)
{
}


RooCategory::~RooCategory()
{
}


RooCategory& RooCategory::operator=(Int_t index) {
  setIndex(index,kTRUE) ;
  return *this ;
}


RooCategory& RooCategory::operator=(const char*label) {
  setLabel(label) ;
  return *this ;
}


RooCategory& RooCategory::operator=(RooCategory& other)
{
  RooAbsCategory::operator=(other) ;
  return *this ;
}


RooAbsArg& RooCategory::operator=(RooAbsArg& aother)
{
  return operator=((RooCategory&)aother) ;
}



Bool_t RooCategory::setIndex(Int_t index, Bool_t printError) 
{
  const RooCatType* type = lookupType(index,printError) ;
  if (!type) return kTRUE ;
  _value = *type ;
  setValueDirty(kTRUE) ;
  return kFALSE ;
}



Bool_t RooCategory::setLabel(const char* label, Bool_t printError) 
{
  const RooCatType* type = lookupType(label,printError) ;
  if (!type) return kTRUE ;
  _value = *type ;
  setValueDirty(kTRUE) ;
  return kFALSE ;
}


Bool_t RooCategory::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  // Read object contents from given stream

  // Read single token
  RooStreamParser parser(is) ;
  TString token = parser.readToken() ;
  return setLabel(token,verbose) ;
}



void RooCategory::writeToStream(ostream& os, Bool_t compact)
{
  // compact only at the moment
  if (compact) {
    os << getIndex() ;
  } else {
    os << getLabel() ;
  }
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

void RooCategory::postTreeLoadHook() 
{
  if (isValid()) {
    // Synchronize label with new index
    _value = *lookupType(_value.getVal()) ;
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



