/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooNameSet.cxx,v 1.21 2007/05/11 09:11:58 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --

#include "RooFit.h"

#include "TObjString.h"
#include "TObjString.h"
#include "RooNameSet.h"
#include "RooArgSet.h"
#include "RooArgList.h"

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



RooNameSet::RooNameSet(const RooNameSet& other) : TObject(other), RooPrintable(other), _nameList()
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
  RooArgList tmp(argSet) ;
  tmp.sort() ;
  TIterator* iter = tmp.createIterator() ;
  RooAbsArg* arg ;
  char *ptr=_nameList ;
  char *end=_nameList+_len-2 ;
  while((arg=(RooAbsArg*)iter->Next())) {    
    const char* argName = arg->GetName() ;
    while((*ptr++ = *argName++)) {
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


RooNameSet& RooNameSet::operator=(const RooNameSet& other) 
{
  delete[] _nameList ;

  _len = other._len ;
  _nameList = new char[_len] ;
  strcpy(_nameList,other._nameList) ;  

  return *this ;
}


void RooNameSet::printToStream(ostream &os, PrintOption /*opt*/, TString indent) const{
  os << indent << _nameList << endl ;
}
