/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooLinkedList.cc,v 1.10 2004/04/05 22:44:12 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
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

#include "RooFitCore/RooLinkedList.hh"
#include "RooFitCore/RooLinkedListIter.hh"
#include "RooFitCore/RooHashTable.hh"
#include "RooFitCore/RooAbsArg.hh"

ClassImp(RooLinkedList)
;

RooLinkedList::RooLinkedList(Int_t htsize) : 
  _size(0), _first(0), _last(0), _htableName(0), _htablePtr(0)
{
  setHashTableSize(htsize) ;
}




RooLinkedList::RooLinkedList(const RooLinkedList& other) :
  _size(0), _first(0), _last(0), _htableName(0), _htablePtr(0) 
{
  // Copy constructor

  if (other._htableName) _htableName = new RooHashTable(other._htableName->size()) ;
  if (other._htablePtr)  _htableName = new RooHashTable(other._htablePtr->size()) ;
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
      delete _htablePtr ;
      _htableName = 0 ;
      _htablePtr = 0 ;
    }
  } else {
    // (Re)create hash tables
    if (_htableName) delete _htableName ;
    if (_htablePtr) delete _htablePtr ;
    _htableName = new RooHashTable(size) ;
    _htablePtr = new RooHashTable(size,kTRUE) ;
    
    // Fill hash table with existing entries
    RooLinkedListElem* ptr = _first ;
    while(ptr) {
      // 	cout << "setHashTableSize:: filling arg " << ptr->_arg << endl ;
      _htableName->add(ptr->_arg) ;
      _htablePtr->add(ptr->_arg) ;
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
  if (_htablePtr) {
    delete _htablePtr ;
  }
}


void RooLinkedList::Add(TObject* arg, Int_t refCount)
{
  //     cout << "RLL::Add " << arg->GetName() ;
  if (!arg) return ;
  
  // Add to hash table 
  if (_htableName){
    _htableName->add(arg) ;
    _htablePtr->add(arg) ;
  }

  if (_last) {
    // Append element at end of list
    _last = new RooLinkedListElem(arg,_last) ;
  } else {
    // Append first element, set first,last 
    _last = new RooLinkedListElem(arg) ;
    _first=_last ;
  }
  _size++ ;
  _last->_refCount = refCount ;
  //     cout << "... size now" << _size << endl ;
}




Bool_t RooLinkedList::Remove(TObject* arg) 
{
  // Find link element
  RooLinkedListElem* elem = findLink(arg) ;
  if (!elem) return kFALSE ;
  
  // Remove from hash table
  if (_htableName) {
    _htableName->remove(arg) ;
    _htablePtr->remove(arg) ;
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
    _htablePtr->replace(oldArg,newArg) ;
  }

  elem->_arg = (TObject*)newArg ;
  return kTRUE ;
}


TObject* RooLinkedList::FindObject(const char* name) const 
{
  return find(name) ;
}


// WVE rewrite with ptr hash
TObject* RooLinkedList::FindObject(const TObject* obj) const 
{
  if (_htablePtr) return _htablePtr->find(obj) ;

  RooLinkedListElem *elem = findLink((TObject*)obj) ;
  return elem ? elem->_arg : 0 ;
}


void RooLinkedList::Clear(Option_t *o) 
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
    delete _htablePtr ;
    _htableName = new RooHashTable(hsize) ;   
    _htablePtr = new RooHashTable(hsize,kTRUE) ;   
  }
}



// WVE need to delete hash too?
void RooLinkedList::Delete(Option_t *o) 
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

