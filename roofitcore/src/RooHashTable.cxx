/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooHashTable.cc,v 1.6 2002/09/17 06:39:34 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#include "TMath.h"
#include "TCollection.h"
#include "RooFitCore/RooHashTable.hh"
#include "RooFitCore/RooLinkedList.hh"
#include "RooFitCore/RooAbsArg.hh"

ClassImp(RooHashTable)
;

RooHashTable::RooHashTable(Int_t capacity) 
{
  if (capacity <= 0) {
    capacity = TCollection::kInitHashTableCapacity;
  }  
  _size = (Int_t)TMath::NextPrime(TMath::Max(capacity,(int)TCollection::kInitHashTableCapacity));
  _arr  = new RooLinkedList* [_size] ;
  memset(_arr, 0, _size*sizeof(RooLinkedList*));

  _usedSlots = 0 ;
  _entries   = 0 ;
}


RooHashTable::RooHashTable(const RooHashTable& other) :
  _usedSlots(other._usedSlots), _entries(other._entries), _size(other._size)
{
  _arr  = new RooLinkedList* [_size] ;
  memset(_arr, 0, _size*sizeof(RooLinkedList*));  
  Int_t i ;
  for (i=0 ; i<_size ; i++) {
    if (other._arr[i]) {
      _arr[i] = new RooLinkedList(*other._arr[i]) ;
    }
  }
}


void RooHashTable::add(TObject* arg) 
{
  Int_t slot = arg->Hash() % _size ;
  if (!_arr[slot]) {
    _arr[slot] = new RooLinkedList(kFALSE) ;
    _usedSlots++ ;
   }
   _arr[slot]->Add(arg);
   _entries++;
}



Bool_t RooHashTable::remove(TObject* arg)
{
  Int_t slot = arg->Hash() % _size ;
  if (_arr[slot]) {
    if (_arr[slot]->Remove(arg)) {
      _entries-- ;
      if (_arr[slot]->GetSize()==0) {
	delete _arr[slot] ;
	_arr[slot] = 0 ;
	_usedSlots-- ;
      }
      return kTRUE ;
    }
  }
  return kFALSE ;
}



Double_t RooHashTable::avgCollisions() const 
{
  Int_t i,h[20] ;
  for (i=0 ;  i<20 ; i++) h[i]=0 ; 

  for (i=0 ; i<_size ; i++) {
    if (_arr[i]) {
      Int_t count = _arr[i]->GetSize() ;
      if (count<20) {
	h[count]++ ;
      } else {
	h[19]++ ;
      }
    } else {
      h[0]++ ;
    }
  }

  for (i=0 ; i<20 ; i++) {
    cout << "h[" << i << "] = " << h[i] << endl ;
  }
  return 0 ;
}


Bool_t RooHashTable::replace(const TObject* oldArg, const TObject* newArg) 
{
  Int_t slot = oldArg->Hash() % _size ;
  if (_arr[slot]) {
    return _arr[slot]->Replace(oldArg,newArg) ;
  }
  return kFALSE ;
}


TObject* RooHashTable::find(const char* name) const 
{
  Int_t slot = TString(name).Hash() % _size ;
  if (_arr[slot]) return _arr[slot]->find(name) ;
  return 0;  
}


RooHashTable::~RooHashTable() 
{  
  Int_t i ;
  for (i=0 ; i<_size ; i++) {
    if (_arr[i]) delete _arr[i] ;  
  }
  delete[] _arr ;
}
