/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooNameSet.cc,v 1.2 2001/08/03 02:04:32 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/


#include "TObjString.h"
#include "RooFitCore/RooNameSet.hh"
#include "RooFitCore/RooArgSet.hh"

ClassImp(RooNameSet)
;

RooNameSet::RooNameSet()
{
  _nameIter = _nameList.MakeIterator() ;
}




RooNameSet::RooNameSet(const RooArgSet& argSet)  
{
  refill(argSet) ;
  _nameIter = _nameList.MakeIterator() ;
}



RooNameSet::RooNameSet(const RooNameSet& other) : _nameList()
{
  other._nameIter->Reset() ;
  TObject* obj;
  while(obj=other._nameIter->Next()) {
    _nameList.Add(obj->Clone()) ;
  }
}


void RooNameSet::refill(const RooArgSet& argSet) 
{
  _nameList.Delete() ;
  TIterator* iter = argSet.createIterator() ;
  RooAbsArg* arg ;
  while(arg=(RooAbsArg*)iter->Next()) {    
    _nameList.Add(new TObjString(arg->GetName())) ;
  }
  _nameList.Sort() ;

  delete iter ;
}



RooNameSet::~RooNameSet() 
{
  delete _nameIter ;
  _nameList.Delete() ;
}




RooArgSet* RooNameSet::select(const RooArgSet& input)
{
  RooArgSet* output = new RooArgSet ;
  _nameIter->Reset() ;
  TObjString* str ;
  while(str=(TObjString*)_nameIter->Next()) {
    RooAbsArg* arg = input.find(str->String()) ;
    if (!arg) {
      cout << "RooNameSet::select ERROR: no argument named " 
	   << str->String() << " found in input data set" << endl ;
      continue ;
    }
    output->add(*arg) ;
  }
  return output ;
}

Bool_t RooNameSet::operator==(const RooNameSet& other) 
{
  // Check comparison againt self
  if (&other==this) return kTRUE ;

  // First check for equal length
  if (_nameList.GetSize() != other._nameList.GetSize()) return kFALSE ;

  // Then check for equal contents. Lists are sorted, just need to compare equal slots
  _nameIter->Reset() ;
  other._nameIter->Reset() ;
  TObject *str1, *str2 ;
  while(str1=_nameIter->Next()) {
    str2 = other._nameIter->Next() ;
    if (!str1->IsEqual(str2)) return kFALSE ;
  }

  return kTRUE ;
}


void RooNameSet::printToStream(ostream &os, PrintOption opt, TString indent) const{
  TObjString* str ;
  Bool_t first(kTRUE) ;
  os << indent ;
  _nameIter->Reset() ;
  while(str=(TObjString*)_nameIter->Next()) {
    if (first) { 
      first=kFALSE ;
    } else {
      os << "," ;
    }
    os << str->String() ;
  }
  os << endl ;
}
