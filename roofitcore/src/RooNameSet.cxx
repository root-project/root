/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooNameSet.cc,v 1.8 2001/11/08 19:59:27 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --

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


void RooNameSet::extendBuffer(Int_t inc)
{
  char * newbuf = new char[_len+inc] ;
  strncpy(newbuf,_nameList,_len) ;
  delete[] _nameList ;
  _nameList = newbuf ;
  _len += inc ;
}


void RooNameSet::refill(const RooArgSet& argSet) 
{
  TIterator* iter = argSet.createIterator() ;
  RooAbsArg* arg ;
  char *ptr=_nameList ;
  char *end=_nameList+_len-2 ;
  while(arg=(RooAbsArg*)iter->Next()) {    
    const char* argName = arg->GetName() ;
    while(*ptr++ = *argName++) {
      if (ptr>=end) {
	// Extend buffer
	Int_t offset = ptr-_nameList ;
	extendBuffer(1024) ;
	ptr = _nameList + offset ;
	end = _nameList + _len - 2;
      }
    }
    *(ptr-1) = ':' ;
  }
  if (ptr>_nameList) *(ptr-1)= 0 ;
  delete iter ;
}


RooArgSet* RooNameSet::select(const RooArgSet& list) const 
{
  RooArgSet* output = new RooArgSet ;

  char buffer[1024] ;
  strcpy(buffer,_nameList) ;
  char* token = strtok(buffer,":") ;
  
  while(token) {
    RooAbsArg* arg =  list.find(token) ;
    if (arg) output->add(*arg) ;
    token = strtok(0,":") ;
  }

  return output ;
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
  os << strlen(_nameList) << endl ;
  os << indent << _nameList << endl ;
}
