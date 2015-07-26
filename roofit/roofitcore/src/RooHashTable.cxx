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

#include "RooFit.h"

#include "TMath.h"
#include "TMath.h"
#include "TCollection.h"
#include "RooHashTable.h"
#include "RooLinkedList.h"
#include "RooAbsArg.h"
#include "RooSetPair.h"

using namespace std;

ClassImp(RooHashTable)
;

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// RooHashTable implements a hash table for TObjects. The hashing can be
// done on the object addresses, object names, or using the objects
// internal hash method. This is a utility class for RooLinkedList
// that uses RooHashTable to speed up direct access to large collections.
// END_HTML
//

//_____________________________________________________________________________
RooHashTable::RooHashTable(Int_t capacity, HashMethod hashMethod) :
  _hashMethod(hashMethod)
{
  // Construct a hash table with given capacity and hash method

  if (capacity <= 0) {
    capacity = TCollection::kInitHashTableCapacity;
  }  
  _size = (Int_t)TMath::NextPrime(TMath::Max(capacity,(int)TCollection::kInitHashTableCapacity));
  _arr  = new RooLinkedList* [_size] ;
  memset(_arr, 0, _size*sizeof(RooLinkedList*));

  _usedSlots = 0 ;
  _entries   = 0 ;
}



//_____________________________________________________________________________
RooHashTable::RooHashTable(const RooHashTable& other) :
  TObject(other),
  _hashMethod(other._hashMethod),
  _usedSlots(other._usedSlots), 
  _entries(other._entries), 
  _size(other._size)
{
  // Copy constructor

  _arr  = new RooLinkedList* [_size] ;
  memset(_arr, 0, _size*sizeof(RooLinkedList*));  
  Int_t i ;
  for (i=0 ; i<_size ; i++) {
    if (other._arr[i]) {
      _arr[i] = new RooLinkedList(*other._arr[i]) ;
    }
  }
}



//_____________________________________________________________________________
void RooHashTable::add(TObject* arg, TObject* hashArg) 
{
  // Add given object to table. If hashArg is given, hash will be calculation
  // on that rather than on 'arg'

  Int_t slot = hash(hashArg?hashArg:arg) % _size ;
  if (!_arr[slot]) {
    _arr[slot] = new RooLinkedList(0) ;
    _arr[slot]->useNptr(kFALSE) ;
    _usedSlots++ ;
   }
   _arr[slot]->Add(arg);
   _entries++;
}



//_____________________________________________________________________________
Bool_t RooHashTable::remove(TObject* arg, TObject* hashArg)
{
  // Remove given object from table. If hashArg is given, hash will be calculation
  // on that rather than on 'arg'

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

  if (_hashMethod != Name) return kFALSE;

  // If we didn't find it by name, see if it might have been renamed
  RooAbsArg* p = dynamic_cast<RooAbsArg*>(arg);
  //cout << "RooHashTable::remove possibly renamed '" << arg->GetName() << "', kRenamedArg=" << (p&&p->namePtr()->TestBit(RooNameReg::kRenamedArg)) << endl;
  if (p && !p->namePtr()->TestBit(RooNameReg::kRenamedArg)) return kFALSE;

  // If so, check the whole list
  Int_t i;
  for (i=0 ; i<_size ; i++) {
    if (i != slot && _arr[i] && _arr[i]->Remove(arg)) {
      _entries-- ;
      if (_arr[i]->GetSize()==0) {
        delete _arr[i] ;
        _arr[i] = 0 ;
        _usedSlots-- ;
      }
      return kTRUE ;
    }
  }

  return kFALSE ;
}



//_____________________________________________________________________________
Double_t RooHashTable::avgCollisions() const 
{
  // Calculate the average number of collisions (table slots with >1 filled entry)
  
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

  return 0 ;
}



//_____________________________________________________________________________
Bool_t RooHashTable::replace(const TObject* oldArg, const TObject* newArg, const TObject* oldHashArg) 
{
  // Replace oldArg with newArg in the table. If oldHashArg is given, use that to calculate
  // the hash associated with oldArg

  Int_t slot = hash(oldHashArg?oldHashArg:oldArg) % _size ;
  if (_arr[slot]) {
    Int_t newSlot = hash(newArg) % _size ;
    if (newSlot == slot) {
      return _arr[slot]->Replace(oldArg,newArg) ;
    }
  }

  // We didn't find the oldArg or they have different slots.
  if (remove((TObject*)oldArg,(TObject*)oldHashArg)) {
    add((TObject*)newArg);
    return kTRUE;
  }
  return kFALSE ;
}



//_____________________________________________________________________________
TObject* RooHashTable::find(const char* name) const 
{
  // Return the object with given name from the table.

  if (_hashMethod != Name) assert(0) ;

  Int_t slot = TMath::Hash(name) % _size ;
  if (_arr[slot]) return _arr[slot]->find(name) ;
  return 0;  
}



//_____________________________________________________________________________
RooAbsArg* RooHashTable::findArg(const RooAbsArg* arg) const 
{
  if (_hashMethod != Name) assert(0) ;
  
  Int_t slot = TMath::Hash(arg->GetName()) % _size ;
  if (_arr[slot]) return _arr[slot]->findArg(arg) ;
  return 0;  
}



//_____________________________________________________________________________
TObject* RooHashTable::find(const TObject* hashArg) const 
{
  // Return object with the given pointer from the table

  RooLinkedListElem* elem = findLinkTo(hashArg) ;
  return elem ? elem->_arg : 0 ;
}



//_____________________________________________________________________________
RooLinkedListElem* RooHashTable::findLinkTo(const TObject* hashArg) const 
{
  // Return RooLinkedList element link to object 'hashArg'

  if (_hashMethod != Pointer) assert(0) ;

  Int_t slot = hash(hashArg) % _size ;
  RooLinkedList* lst = _arr[slot];
  if (lst) {
    RooFIter it = lst->fwdIterator() ;
    TObject* obj;
    while ((obj=it.next())) {
      RooLinkedListElem* elem = (RooLinkedListElem*)obj ;
      if (elem->_arg == hashArg) return elem ;
    }
  }
  return 0;  
}



//_____________________________________________________________________________
RooSetPair* RooHashTable::findSetPair(const RooArgSet* set1, const RooArgSet* set2) const 
{  
  // Return RooSetPair with given pointers in table

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




//_____________________________________________________________________________
RooHashTable::~RooHashTable() 
{  
  // Destructor

  Int_t i ;
  for (i=0 ; i<_size ; i++) {
    if (_arr[i]) delete _arr[i] ;  
  }
  delete[] _arr ;
}
