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
// RooLinkedList is an collection class for internal use, storing
// a collection of RooAbsArg pointers in a doubly linked list.
// It can optionally add a hash table to speed up random access
// in large collections
// Use RooAbsCollection derived objects for public use
// (e.g. RooArgSet or RooArgList) 
// END_HTML
//

#include "RooFit.h"
#include "Riostream.h"

#include "RooLinkedList.h"
#include "RooLinkedListIter.h"
#include "RooHashTable.h"
#include "RooAbsArg.h"
#include "RooMsgService.h"



ClassImp(RooLinkedList)
;


//_____________________________________________________________________________
RooLinkedList::RooLinkedList(Int_t htsize) : 
  _hashThresh(htsize), _size(0), _first(0), _last(0), _htableName(0), _htableLink(0)
{
  // Constructor with hashing threshold. If collection size exceeds threshold
  // a hash table is added.
//   if (htsize>0) {
//     cout << "RooLinkedList::ctor htsize=" << htsize << endl ;
//   }
}




//_____________________________________________________________________________
RooLinkedList::RooLinkedList(const RooLinkedList& other) :
  TObject(other), _hashThresh(other._hashThresh), _size(0), _first(0), _last(0), _htableName(0), _htableLink(0), _name(other._name)
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



//_____________________________________________________________________________
RooLinkedList& RooLinkedList::operator=(const RooLinkedList& other) 
{
  // Assignment operator, copy contents from 'other'
  
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



//_____________________________________________________________________________
void RooLinkedList::setHashTableSize(Int_t size) 
{        
  // Change the threshold for hash-table use to given size.
  // If a hash table exists when this method is called, it is regenerated.

  if (size<0) {
    coutE(InputArguments) << "RooLinkedList::setHashTable() ERROR size must be positive" << endl ;
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


 

//_____________________________________________________________________________
RooLinkedList::~RooLinkedList() 
{
  // Destructor

  if (_htableName) {
    delete _htableName ;
    _htableName=0 ;
  }
  if (_htableLink) {
    delete _htableLink ;
    _htableLink=0 ;
  }

  Clear() ;
}



//_____________________________________________________________________________
RooLinkedListElem* RooLinkedList::findLink(const TObject* arg) const 
{    
  // Find the element link containing the given object

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



//_____________________________________________________________________________
void RooLinkedList::Add(TObject* arg, Int_t refCount)
{
  // Insert object into collection with given reference count value

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




//_____________________________________________________________________________
Bool_t RooLinkedList::Remove(TObject* arg) 
{
  // Remove object from collection

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




//_____________________________________________________________________________
TObject* RooLinkedList::At(Int_t index) const 
{
  // Return object stored in sequential position given by index.
  // If index is out of range, a null pointer is returned.

  // Check range
  if (index<0 || index>=_size) return 0 ;

  
  // Walk list
  RooLinkedListElem* ptr = _first;
  while(index--) ptr = ptr->_next ;
  
  // Return arg
  return ptr->_arg ;
}




//_____________________________________________________________________________
Bool_t RooLinkedList::Replace(const TObject* oldArg, const TObject* newArg) 
{
  // Replace object 'oldArg' in collection with new object 'newArg'.
  // If 'oldArg' is not found in collection kFALSE is returned

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



//_____________________________________________________________________________
TObject* RooLinkedList::FindObject(const char* name) const 
{
  // Return pointer to obejct with given name. If no such object
  // is found return a null pointer

  return find(name) ;
}



//_____________________________________________________________________________
TObject* RooLinkedList::FindObject(const TObject* obj) const 
{
  // Find object in list. If list contains object return 
  // (same) pointer to object, otherwise return null pointer

  RooLinkedListElem *elem = findLink((TObject*)obj) ;
  return elem ? elem->_arg : 0 ;
}



//_____________________________________________________________________________
void RooLinkedList::Clear(Option_t *) 
{
  // Remove all elements from collection

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



//_____________________________________________________________________________
void RooLinkedList::Delete(Option_t *) 
{
  // Remove all elements in collection and delete all elements
  // NB: Collection does not own elements, this function should
  // be used judiciously by caller. 

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


  

//_____________________________________________________________________________
TObject* RooLinkedList::find(const char* name) const 
{
  // Return pointer to object with given name in collection.
  // If no such object is found, return null pointer.

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



//_____________________________________________________________________________
Int_t RooLinkedList::IndexOf(const TObject* arg) const 
{
  // Return position of given object in list. If object
  // is not contained in list, return -1

  RooLinkedListElem* ptr = _first;
  Int_t idx(0) ;
  while(ptr) {
    if (ptr->_arg==arg) return idx ;
    ptr = ptr->_next ;
    idx++ ;
  }
  return -1 ;
}



//_____________________________________________________________________________
Int_t RooLinkedList::IndexOf(const char* name) const 
{
  // Return position of given object in list. If object
  // is not contained in list, return -1

  RooLinkedListElem* ptr = _first;
  Int_t idx(0) ;
  while(ptr) {
    if (strcmp(ptr->_arg->GetName(),name)==0) return idx ;
    ptr = ptr->_next ;
    idx++ ;
  }
  return -1 ;
}



//_____________________________________________________________________________
void RooLinkedList::Print(const char* opt) const 
{
  // Print contents of list, defers to Print() function
  // of contained objects
  RooLinkedListElem* elem = _first ;
  while(elem) {
    cout << elem->_arg << " : " ; 
    elem->_arg->Print(opt) ;
    elem = elem->_next ;
  }    
}


//_____________________________________________________________________________
RooLinkedListIter RooLinkedList::iterator(Bool_t dir) const 
{
  return RooLinkedListIter(this,dir) ;
}


//_____________________________________________________________________________
RooFIter RooLinkedList::fwdIterator() const 
{ 
  return RooFIter(this) ; 
}


//_____________________________________________________________________________
TIterator* RooLinkedList::MakeIterator(Bool_t dir) const 
{
  // Return an iterator over this list
  return new RooLinkedListIter(this,dir) ;
}




//_____________________________________________________________________________
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



//_____________________________________________________________________________
void RooLinkedList::swapWithNext(RooLinkedListElem* elemB) 
{
  // Swap given to elements in the linked list. Auxiliary function for Sort()

  RooLinkedListElem* elemA = elemB->_prev ;
  RooLinkedListElem* elemC = elemB->_next ;
  if (!elemC) return ;

  RooLinkedListElem* elemD = elemC->_next ;

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



// void Roo1DTable::Streamer(TBuffer &R__b)
// {
//    // Stream an object of class Roo1DTable.

//    if (R__b.IsReading()) {
//       R__b.ReadClassBuffer(Roo1DTable::Class(),this);
//    } else {
//       R__b.WriteClassBuffer(Roo1DTable::Class(),this);
//    }
// }



//_____________________________________________________________________________
void RooLinkedList::Streamer(TBuffer &R__b)
{
  // Custom streaming handling schema evolution w.r.t past implementations

  if (R__b.IsReading()) {

    Version_t v = R__b.ReadVersion();
    //R__b.ReadVersion();
    TObject::Streamer(R__b);

    Int_t size ;
    TObject* arg ;

    R__b >> size ;
    while(size--) {
      R__b >> arg ;
      Add(arg) ;      
    }

    if (v>1) {
      R__b >> _name ;
    }

  } else {
    R__b.WriteVersion(RooLinkedList::IsA());
    TObject::Streamer(R__b);
    R__b << _size ;

    RooLinkedListElem* ptr = _first;
    while(ptr) {
      R__b << ptr->_arg ;
      ptr = ptr->_next ;
    } 

    R__b << _name ;
  }
}

