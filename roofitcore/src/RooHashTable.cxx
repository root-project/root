/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooHashTable.cxx,v 1.17 2007/05/11 09:11:58 verkerke Exp $
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

#include "RooFit.h"

#include "TMath.h"
#include "TMath.h"
#include "TCollection.h"
#include "RooHashTable.h"
#include "RooLinkedList.h"
#include "RooAbsArg.h"
#include "RooSetPair.h"

ClassImp(RooHashTable)
;

RooHashTable::RooHashTable(Int_t capacity, HashMethod hashMethod) :
  _hashMethod(hashMethod)
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
  TObject(other),
  _hashMethod(other._hashMethod),
  _usedSlots(other._usedSlots), 
  _entries(other._entries), 
  _size(other._size)
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


void RooHashTable::add(TObject* arg, TObject* hashArg) 
{
  Int_t slot = hash(hashArg?hashArg:arg) % _size ;
  if (!_arr[slot]) {
    _arr[slot] = new RooLinkedList(0) ;
    _usedSlots++ ;
   }
   _arr[slot]->Add(arg);
   _entries++;
}



Bool_t RooHashTable::remove(TObject* arg, TObject* hashArg)
{
  Int_t slot = hash(hashArg?hashArg:arg) % _size ;
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


Bool_t RooHashTable::replace(const TObject* oldArg, const TObject* newArg, const TObject* oldHashArg) 
{
  Int_t slot = hash(oldHashArg?oldHashArg:oldArg) % _size ;
  if (_arr[slot]) {
    return _arr[slot]->Replace(oldArg,newArg) ;
  }
  return kFALSE ;
}


TObject* RooHashTable::find(const char* name) const 
{
  if (_hashMethod != Name) assert(0) ;

  Int_t slot = TMath::Hash(name) % _size ;
  if (_arr[slot]) return _arr[slot]->find(name) ;
  return 0;  
}


TObject* RooHashTable::find(const TObject* hashArg) const 
{
  if (_hashMethod != Pointer) assert(0) ;

  Int_t slot = hash(hashArg) % _size ;
  if (_arr[slot]) return _arr[slot]->FindObject(hashArg) ;
  return 0;  
}


RooLinkedListElem* RooHashTable::findLinkTo(const TObject* hashArg) const 
{
  if (_hashMethod != Pointer) assert(0) ;

  Int_t slot = hash(hashArg) % _size ;
  if (_arr[slot]) {
    Int_t i ; 
    for (i=0 ; i<_arr[slot]->GetSize() ; i++) {
      RooLinkedListElem* elem = (RooLinkedListElem*)_arr[slot]->At(i) ;
      if (elem->_arg == hashArg) return elem ;
    }
  }
  return 0;  
}



RooSetPair* RooHashTable::findSetPair(const RooArgSet* set1, const RooArgSet* set2) const 
{  
  if (_hashMethod != Intrinsic) assert(0) ;

  Int_t slot = RooSetPair(set1,set2).Hash() % _size ;
  if (_arr[slot]) {
    Int_t i ; 
    for (i=0 ; i<_arr[slot]->GetSize() ; i++) {
      RooSetPair* pair = (RooSetPair*)_arr[slot]->At(i) ;
      if (pair->_set1==set1 && pair->_set2==set2) {
	return pair ;
      }
    }
  }

  return 0 ;
}




RooHashTable::~RooHashTable() 
{  
  Int_t i ;
  for (i=0 ; i<_size ; i++) {
    if (_arr[i]) delete _arr[i] ;  
  }
  delete[] _arr ;
}
