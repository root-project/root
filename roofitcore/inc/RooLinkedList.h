/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooLinkedList.rdl,v 1.1 2001/10/17 05:03:59 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   16-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_LINKED_LIST
#define ROO_LINKED_LIST

#include "TObject.h"
#include "RooFitCore/RooLinkedListElem.hh"
class RooLinkedListIter ;

class RooLinkedList : public TObject {
public:
  // Constructor
  RooLinkedList() : 
    _size(0), _first(0), _last(0) {
  }

  // Copy constructor
  RooLinkedList(const RooLinkedList& other) :
    _size(0), _first(0), _last(0) {
    RooLinkedListElem* elem = other._first ;
    while(elem) {
      Add(elem->_arg) ;
      elem = elem->_next ;
    }
  }


  // Destructor
  virtual ~RooLinkedList() {
    Clear() ;
  }

  Int_t GetSize() const { return _size ; }

  void Add(RooAbsArg* arg) {
    if (!arg) return ;

    if (_last) {
      // Append element at end of list
      _last = new RooLinkedListElem(arg,_last) ;
    } else {
      // Append first element, set first,last 
      _last = new RooLinkedListElem(arg) ;
      _first=_last ;
    }
    _size++ ;
  }


  Bool_t Remove(RooAbsArg* arg) {
    // Find link element
    RooLinkedListElem* elem = findLink(arg) ;
    if (!elem) return kFALSE ;

    // Update first,last if necessary
    if (elem==_first) _first=elem->_next ;
    if (elem==_last) _last=elem->_prev ;

    // Delete and shrink
    _size-- ;
    delete elem ;	
  }

  RooAbsArg* At(Int_t index) const {
    // Check range
    if (index<0 || index>=_size) return 0 ;

    // Walk list
    RooLinkedListElem* ptr(_first) ;
    while(index--) ptr = ptr->_next ;

    // Return arg
    return ptr->_arg ;
  }


  Bool_t Replace(const RooAbsArg* oldArg, const RooAbsArg* newArg) {
    // Find existing element and replace arg
    RooLinkedListElem* elem = findLink(oldArg) ;
    if (!elem) return kFALSE ;
    elem->_arg = (RooAbsArg*)newArg ;
  }

  TIterator* MakeIterator(Bool_t dir) const ;

  void Clear(Option_t *o=0) {
    RooLinkedListElem* elem(_first) ;
    while(elem) {
      RooLinkedListElem* next = elem->_next ;
      delete elem ;
      elem = next ;
    }
    _first = 0 ;
    _last = 0 ;
    _size = 0 ;
  }

  void Delete(Option_t *o=0) {
    RooLinkedListElem* elem(_first) ;
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
  
  RooAbsArg* find(const char* name) const {
    RooLinkedListElem* ptr(_first) ;
    while(ptr) {
      if (!strcmp(ptr->_arg->GetName(),name)) {
	return ptr->_arg ;
      }
      ptr = ptr->_next ;
    }
    return 0 ;
  }

  Int_t IndexOf(const RooAbsArg* arg) const {
    RooLinkedListElem* ptr(_first) ;
    Int_t idx(0) ;
    while(ptr) {
      if (ptr->_arg==arg) return idx ;
      ptr = ptr->_next ;
      idx++ ;
    }
    return -1 ;
  }

  RooAbsArg* First() const {
    return _first?_first->_arg:0 ;
  }

  void Print(const char* opt) const {
    RooLinkedListElem* elem = _first ;
    while(elem) {
      cout << elem << " : " ; 
      elem->_arg->Print(opt) ;
      elem = elem->_next ;
    }    
  }

  void Sort(Bool_t ascend) ;

protected:  

  friend class RooLinkedListIter ;

  void swapWithNext(RooLinkedListElem* elem) ;

  RooLinkedListElem* findLink(const RooAbsArg* arg) {    
    RooLinkedListElem* ptr(_first) ;
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

  ClassDef(RooLinkedList,1) // TList with extra support for Option_t associations
};




#endif
