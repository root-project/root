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
#include "RooFitCore/RooAbsCategoryLValue.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooStreamParser.hh"

ClassImp(RooAbsCategoryLValue) 
;


RooAbsCategoryLValue::RooAbsCategoryLValue(const char *name, const char *title) : 
  RooAbsCategory(name,title)
{
  setValueDirty(kTRUE) ;  
  setShapeDirty(kTRUE) ;  
}


RooAbsCategoryLValue::RooAbsCategoryLValue(const RooAbsCategoryLValue& other, const char* name) :
  RooAbsCategory(other, name)
{
}


RooAbsCategoryLValue::~RooAbsCategoryLValue()
{
}


RooAbsCategoryLValue& RooAbsCategoryLValue::operator=(Int_t index) {
  setIndex(index,kTRUE) ;
  return *this ;
}


RooAbsCategoryLValue& RooAbsCategoryLValue::operator=(const char*label) {
  setLabel(label) ;
  return *this ;
}


RooAbsCategoryLValue& RooAbsCategoryLValue::operator=(const RooAbsCategoryLValue& other)
{
  RooAbsCategory::operator=(other) ;
  return *this ;
}


RooAbsArg& RooAbsCategoryLValue::operator=(const RooAbsArg& aother)
{
  return operator=((const RooAbsCategoryLValue&)aother) ;
}



Bool_t RooAbsCategoryLValue::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  // Read object contents from given stream
}



void RooAbsCategoryLValue::writeToStream(ostream& os, Bool_t compact) const
{
}



void RooAbsCategoryLValue::attachToTree(TTree& t, Int_t bufSize)
{
  // Attach object to a branch of given TTree

  // First determine if branch is taken
  if (t.GetBranch(GetName())) {
    cout << "RooAbsCategoryLValue::attachToTree(" << GetName() << "): branch in tree " << t.GetName() << " already exists" << endl ;
    t.SetBranchAddress(GetName(),&((Int_t&)_value)) ;
  } else {    
    TString format(GetName());
    format.Append("/I");
    void* ptr = &(_value._value) ;
    //    _value.setVal(999) ;
    cout << "RooAbsCategoryLValue::attachToTree(" << GetName() << "): making new branch in tree " << t.GetName() 
	 << ", prt=" << ptr  << endl ;    
    t.Branch(GetName(), ptr, (const Text_t*)format, bufSize);
  }
}

void RooAbsCategoryLValue::postTreeLoadHook() 
{
  if (isValid()) {
    // Synchronize label with new index
    _value = *lookupType(_value.getVal()) ;
  }
}




