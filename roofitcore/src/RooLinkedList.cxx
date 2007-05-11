/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooLinkedList.cc,v 1.18 2005/06/20 15:44:54 wverkerke Exp $
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

// -- CLASS DESCRIPTION [AUX] ---
// RooLinkedList is an collection class for internal use, storing
// a collection of RooAbsArg pointers in a doubly linked list
// Use RooAbsCollection derived objects for public use
// (RooArgSet and RooArgList) 

#include "RooFit.h"

#include "RooLinkedList.h"
#include "RooLinkedList.h"
#include "RooLinkedListIter.h"
#include "RooHashTable.h"
#include "RooAbsArg.h"

ClassImp(RooLinkedList)
;

RooLinkedList::RooLinkedList(Int_t htsize) : 
  _hashThresh(htsize), _size(0), _first(0), _last(0), _htableName(0), _htableLink(0)
{
  //setHashTableSize(htsize) ;
}




RooLinkedList::RooLinkedList(const RooLinkedList& other) :
   TObject(other), _hashThresh(other._hashThresh), _size(0), _first(0), _last(0), _htableName(0), _htableLink(0)
{
  // Copy constructor

  if (other._htableName) _htableName = new RooHashTable(other._htableName->size()) ;
  if (other._htableLink) _htableLink = new RooHashTable(other._htableLink->size(),RooHashTable::Pointer) ;
  RooLinkedListElem* elem = other._first ;
  while(elem) {
    Add(elem->_arg, elem->_refCount) ;
    elem = elem->_next ;
  }
}



RooLinkedList& RooLinkedList::operator=(const RooLinkedList& other) 
{
  // Assignment operator
  
  // Prevent self-assignment
  if (&other==this) return *this ;
  
  // Copy elements
  RooLinkedListElem* elem = other._first ;
  while(elem) {
    Add(elem->_arg) ;
    elem = elem->_next ;
  }    
  
  return *this ;
}



void RooLinkedList::setHashTableSize(Int_t size) 
{        
  if (size<0) {
    cout << "RooLinkedList::setHashTable() ERROR size must be positive" << endl ;
    return ;
  }
  if (size==0) {
    if (!_htableName) {
      // No hash table present
      return ;
    } else {
      // Remove existing hash table
      delete _htableName ;
      delete _htableLink ;
      _htableName = 0 ;
      _htableLink = 0 ;
    }
  } else {

    // (Re)create hash tables
    if (_htableName) delete _htableName ;
    _htableName = new RooHashTable(size) ;

     if (_htableLink) delete _htableLink ;
     _htableLink = new RooHashTable(size,RooHashTable::Pointer) ;
    
    // Fill hash table with existing entries
    RooLinkedListElem* ptr = _first ;
    while(ptr) {
      _htableName->add(ptr->_arg) ;
      _htableLink->add((TObject*)ptr,ptr->_arg) ;
      ptr = ptr->_next ;
    }      
  }
}


 
RooLinkedList::~RooLinkedList() 
{
  // Destructor
  Clear() ;
  if (_htableName) {
    delete _htableName ;
  }
  if (_htableLink) {
    delete _htableLink ;
  }
}


RooLinkedListElem* RooLinkedList::findLink(const TObject* arg) const 
{    

  if (_htableLink) {
    return _htableLink->findLinkTo(arg) ;  
  }
  
  RooLinkedListElem* ptr = _first;
  while(ptr) {
    if (ptr->_arg == arg) {
      return ptr ;
    }
    ptr = ptr->_next ;
  }
  return 0 ;
  
}


void RooLinkedList::Add(TObject* arg, Int_t refCount)
{
  if (!arg) return ;
  
  // Add to hash table 
  if (_htableName) {

    // Expand capacity of hash table if #entries>#slots
    if (_size > _htableName->size()) {
      setHashTableSize(_size*2) ;
    }

  } else if (_hashThresh>0 && _size>_hashThresh) {

    setHashTableSize(_hashThresh) ;
  }  

  if (_last) {
    // Append element at end of list
    _last = new RooLinkedListElem(arg,_last) ;
  } else {
    // Append first element, set first,last 
    _last = new RooLinkedListElem(arg) ;
    _first=_last ;
  }

  if (_htableName){
    //cout << "storing link " << _last << " with hash arg " << arg << endl ;
    _htableName->add(arg) ;
    _htableLink->add((TObject*)_last,arg) ;
  }

  _size++ ;
  _last->_refCount = refCount ;
  
}




Bool_t RooLinkedList::Remove(TObject* arg) 
{
  // Find link element
  RooLinkedListElem* elem = findLink(arg) ;
  if (!elem) return kFALSE ;
  
  // Remove from hash table
  if (_htableName) {
    _htableName->remove(arg) ;
  }
  if (_htableLink) {
    _htableLink->remove((TObject*)elem,arg) ;
  }
  
  // Update first,last if necessary
  if (elem==_first) _first=elem->_next ;
  if (elem==_last) _last=elem->_prev ;
  
  // Delete and shrink
  _size-- ;
  delete elem ;	
  return kTRUE ;
}




TObject* RooLinkedList::At(Int_t index) const 
{
  // Check range
  if (index<0 || index>=_size) return 0 ;

//   if (index>10) {
//     cout << "RLL::At(" << index << ")" << endl ;
//   }
  
  // Walk list
  RooLinkedListElem* ptr = _first;
  while(index--) ptr = ptr->_next ;
  
  // Return arg
  return ptr->_arg ;
}




Bool_t RooLinkedList::Replace(const TObject* oldArg, const TObject* newArg) 
{
  // Find existing element and replace arg
  RooLinkedListElem* elem = findLink(oldArg) ;
  if (!elem) return kFALSE ;
  
  if (_htableName) {
    _htableName->replace(oldArg,newArg) ;
  }
  if (_htableLink) {
    // Link is hashed by contents and may change slot in hash table
    _htableLink->remove((TObject*)elem,(TObject*)oldArg) ;
    _htableLink->add((TObject*)elem,(TObject*)newArg) ;
  }

  elem->_arg = (TObject*)newArg ;
  return kTRUE ;
}


TObject* RooLinkedList::FindObject(const char* name) const 
{
  return find(name) ;
}


TObject* RooLinkedList::FindObject(const TObject* obj) const 
{
  RooLinkedListElem *elem = findLink((TObject*)obj) ;
  return elem ? elem->_arg : 0 ;
}


void RooLinkedList::Clear(Option_t *) 
{
  RooLinkedListElem* elem = _first;
  while(elem) {
    RooLinkedListElem* next = elem->_next ;
      delete elem ;
      elem = next ;
  }
  _first = 0 ;
  _last = 0 ;
  _size = 0 ;
  
  if (_htableName) {
    Int_t hsize = _htableName->size() ;
    delete _htableName ;
    _htableName = new RooHashTable(hsize) ;   
  }
  if (_htableLink) {
    Int_t hsize = _htableLink->size() ;
    delete _htableLink ;
    _htableLink = new RooHashTable(hsize,RooHashTable::Pointer) ;       
  }
}



void RooLinkedList::Delete(Option_t *) 
{
  RooLinkedListElem* elem = _first;
  while(elem) {
    RooLinkedListElem* next = elem->_next ;
    delete elem->_arg ;
    delete elem ;
    elem = next ;
  }
  _first = 0 ;
  _last = 0 ;
  _size = 0 ;

  if (_htableName) {
    Int_t hsize = _htableName->size() ;
    delete _htableName ;
    _htableName = new RooHashTable(hsize) ;   
  }
  if (_htableLink) {
    Int_t hsize = _htableLink->size() ;
    delete _htableLink ;
    _htableLink = new RooHashTable(hsize,RooHashTable::Pointer) ;       
  }
}


  
TObject* RooLinkedList::find(const char* name) const 
{
  if (_htableName) return _htableName->find(name) ;

  RooLinkedListElem* ptr = _first ;
  while(ptr) {
    if (!strcmp(ptr->_arg->GetName(),name)) {
      return ptr->_arg ;
    }
    ptr = ptr->_next ;
  }
  return 0 ;
}



Int_t RooLinkedList::IndexOf(const TObject* arg) const 
{
  RooLinkedListElem* ptr = _first;
  Int_t idx(0) ;
  while(ptr) {
    if (ptr->_arg==arg) return idx ;
    ptr = ptr->_next ;
    idx++ ;
  }
  return -1 ;
}



void RooLinkedList::Print(const char* opt) const 
{
  RooLinkedListElem* elem = _first ;
  while(elem) {
    cout << elem->_arg << " : " ; 
    elem->_arg->Print(opt) ;
    elem = elem->_next ;
  }    
}


TIterator* RooLinkedList::MakeIterator(Bool_t dir) const {
  // Return an iterator over this list
  return new RooLinkedListIter(this,dir) ;
}

RooLinkedListIter RooLinkedList::iterator(Bool_t dir) const {
  return RooLinkedListIter(this,dir) ;
}


void RooLinkedList::Sort(Bool_t ascend) 
{
  // Sort elements of this list according to their
  // TObject::Compare() ranking via a simple
  // bubble sort algorithm

  if (_size<2) return ;

  Bool_t working(kTRUE) ;
  while(working) {
    working = kFALSE ;
    RooLinkedListElem* ptr = _first;
    while(ptr && ptr->_next) {    
      if ((ascend && ptr->_arg->Compare(ptr->_next->_arg)>0) ||
	  (!ascend && ptr->_arg->Compare(ptr->_next->_arg)<0)) {
	swapWithNext(ptr) ;
	working = kTRUE ;
      }
      ptr = ptr->_next ;
    }
  }

  return ;
}


void RooLinkedList::swapWithNext(RooLinkedListElem* elemB) 
{
  // Swap given to elements in the linked list. Auxiliary function for Sort()
  RooLinkedListElem* elemA = elemB->_prev ;
  RooLinkedListElem* elemC = elemB->_next ;
  RooLinkedListElem* elemD = elemC->_next ;
  if (!elemC) return ;

  if (elemA) {
    elemA->_next = elemC ;
  } else {
    _first = elemC ;
  }
  elemB->_prev = elemC ;
  elemB->_next = elemD ;
  elemC->_prev = elemA ;
  elemC->_next = elemB ;
  if (elemD) {
    elemD->_prev = elemB ;
  } else {
    _last = elemB ;
  }
  return ;
}


void RooLinkedList::Streamer(TBuffer &b)
{
  if (b.IsReading()) {
    //Version_t v = b.ReadVersion();
    b.ReadVersion();
    TObject::Streamer(b);

    Int_t size ;
    TObject* arg ;

    b >> size ;
    while(size--) {
      b >> arg ;
      Add(arg) ;      
    }

  } else {
    b.WriteVersion(RooLinkedList::IsA());
    TObject::Streamer(b);
    b << _size ;

    RooLinkedListElem* ptr = _first;
    while(ptr) {
      b << ptr->_arg ;
      ptr = ptr->_next ;
    } 
  }
}

