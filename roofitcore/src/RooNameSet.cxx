/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooNameSet.cc,v 1.4 2001/09/17 18:48:15 verkerke Exp $
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
  _len = 1024 ;
  _nameList = new char[_len] ;
  _nameList[0] = 0 ;
  
}




RooNameSet::RooNameSet(const RooArgSet& argSet)  
{
  _len = 1024 ;
  _nameList = new char[_len] ;
  _nameList[0] = 0 ;
  refill(argSet) ;
}



RooNameSet::RooNameSet(const RooNameSet& other) : _nameList()
{
  _len = other._len ;
  _nameList = new char[_len] ;
  strcpy(_nameList,other._nameList) ;
}


void RooNameSet::refill(const RooArgSet& argSet) 
{
  //Make sorted copy of set
//   RooArgSet tmp(argSet) ;
//   tmp.Sort() ;

  TIterator* iter = argSet.createIterator() ;
  RooAbsArg* arg ;
  char *ptr=_nameList ;
  while(arg=(RooAbsArg*)iter->Next()) {    
    const char* argName = arg->GetName() ;
    while(*ptr++ = *argName++) ;
    *ptr++ = ':' ;
  }
  *ptr= 0 ;
  delete iter ;
}


RooNameSet::~RooNameSet() 
{
  delete[] _nameList ;
}


Bool_t RooNameSet::operator==(const RooNameSet& other) 
{
  // Check comparison against self
  if (&other==this) return kTRUE ;

  // First check for equal length
  if (strlen(_nameList) != strlen(other._nameList)) return kFALSE ;

  return (!strcmp(_nameList,other._nameList)) ;
}


void RooNameSet::printToStream(ostream &os, PrintOption opt, TString indent) const{
  TObjString* str ;
  Bool_t first(kTRUE) ;
  os << indent << _nameList << endl ;
}
