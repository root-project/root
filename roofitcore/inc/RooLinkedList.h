/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooLinkedList.rdl,v 1.9 2002/09/05 04:33:37 verkerke Exp $
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
#ifndef ROO_LINKED_LIST
#define ROO_LINKED_LIST

#include "TObject.h"
#include "RooFitCore/RooLinkedListElem.hh"
#include "RooFitCore/RooHashTable.hh"
class TIterator ;
class RooLinkedListIter ;

class RooLinkedList : public TObject {
public:
  // Constructor
  RooLinkedList(Int_t htsize=0) ;

  // Copy constructor
  RooLinkedList(const RooLinkedList& other) ;

  // Assignment operator
  RooLinkedList& operator=(const RooLinkedList& other) ;

  Int_t getHashTableSize() const {
    return _htable ? _htable->size() : 0 ;
  }

  void setHashTableSize(Int_t size) ;

  // Destructor
  virtual ~RooLinkedList() ;

  Int_t GetSize() const { return _size ; }

  virtual void Add(TObject* arg) { Add(arg,1) ; }
  virtual Bool_t Remove(TObject* arg) ;
  TObject* At(Int_t index) const ;
  Bool_t Replace(const TObject* oldArg, const TObject* newArg) ;
  TIterator* MakeIterator(Bool_t dir=kTRUE) const ;
  void Clear(Option_t *o=0) ;
  void Delete(Option_t *o=0) ;
  TObject* find(const char* name) const ;
  TObject* FindObject(const char* name) const ; 
  TObject* FindObject(const TObject* obj) const ;
  Int_t IndexOf(const TObject* arg) const ;
  TObject* First() const {
    return _first?_first->_arg:0 ;
  }

  void Print(const char* opt) const ;
  void Sort(Bool_t ascend) ;

protected:  

  friend class RooLinkedListIter ;

  virtual void Add(TObject* arg, Int_t refCount) ;

  void swapWithNext(RooLinkedListElem* elem) ;

  RooLinkedListElem* findLink(const TObject* arg) const {    
    RooLinkedListElem* ptr = _first;
    while(ptr) {
      if (ptr->_arg == arg) {
	return ptr ;
      }
      ptr = ptr->_next ;
    }
    return 0 ;
  }
    
  Int_t _size ;                //  Current size of list
  RooLinkedListElem*  _first ; //! Link to first element of list
  RooLinkedListElem*  _last ;  //! Link to last element of list
  RooHashTable*       _htable ; //! Hash table 

  ClassDef(RooLinkedList,1) // TList with extra support for Option_t associations
};




#endif
