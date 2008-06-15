/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// RooNameSet is a utility class that stores the names the objects
// in a RooArget. This allows to preserve the contents of a RooArgSet
// in a specific use contents beyond the lifespan of the object in
// the RooArgSet. A new RooArgSet can be created from a RooNameSet
// by offering it a list of new RooAbsArg objects. 
// END_HTML
//

#include "RooFit.h"
#include "Riostream.h"

#include "TObjString.h"
#include "TClass.h"
#include "RooNameSet.h"
#include "RooArgSet.h"
#include "RooArgList.h"



ClassImp(RooNameSet)
;


//_____________________________________________________________________________
RooNameSet::RooNameSet()
{
  _len = 1024 ;
  _nameList = new char[_len] ;
  _nameList[0] = 0 ;
  
}




//_____________________________________________________________________________
RooNameSet::RooNameSet(const RooArgSet& argSet)
{
  _len = 1024 ;
  _nameList = new char[_len] ;
  _nameList[0] = 0 ;
  refill(argSet) ;
}



//_____________________________________________________________________________
RooNameSet::RooNameSet(const RooNameSet& other) : TObject(other), RooPrintable(other), _nameList()
{
  _len = other._len ;
  _nameList = new char[_len] ;
  strcpy(_nameList,other._nameList) ;
}



//_____________________________________________________________________________
void RooNameSet::extendBuffer(Int_t inc)
{
  char * newbuf = new char[_len+inc] ;
  strncpy(newbuf,_nameList,_len) ;
  delete[] _nameList ;
  _nameList = newbuf ;
  _len += inc ;
}



//_____________________________________________________________________________
void RooNameSet::refill(const RooArgSet& argSet) 
{
  RooArgList tmp(argSet) ;
  tmp.sort() ;
  TIterator* iter = tmp.createIterator() ;
  RooAbsArg* arg ;
  char *ptr=_nameList ;
  char *end=_nameList+_len-2 ;
  *ptr = 0 ;
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



//_____________________________________________________________________________
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



//_____________________________________________________________________________
RooNameSet::~RooNameSet() 
{
  delete[] _nameList ;
}



//_____________________________________________________________________________
Bool_t RooNameSet::operator==(const RooNameSet& other) 
{
  // Check comparison against self
  if (&other==this) return kTRUE ;

  // First check for equal length
  if (strlen(_nameList) != strlen(other._nameList)) return kFALSE ;

  return (!strcmp(_nameList,other._nameList)) ;
}



//_____________________________________________________________________________
RooNameSet& RooNameSet::operator=(const RooNameSet& other) 
{
  delete[] _nameList ;

  _len = other._len ;
  _nameList = new char[_len] ;
  strcpy(_nameList,other._nameList) ;  

  return *this ;
}


//_____________________________________________________________________________
void RooNameSet::printName(ostream& os) const 
{
  os << GetName() ;
}


//_____________________________________________________________________________
void RooNameSet::printTitle(ostream& os) const 
{
  os << GetTitle() ;
}


//_____________________________________________________________________________
void RooNameSet::printClassName(ostream& os) const 
{
  os << IsA()->GetName() ;
}


//_____________________________________________________________________________
void RooNameSet::printValue(ostream& os) const 
{
  os << _nameList ;
}
