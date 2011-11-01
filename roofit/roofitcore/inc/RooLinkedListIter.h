/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooLinkedListIter.h,v 1.11 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_LINKED_LIST_ITER
#define ROO_LINKED_LIST_ITER

#include "Rtypes.h"
#include "TIterator.h"
#include "RooLinkedList.h"

#if ROOT_VERSION_CODE <= 332546
#ifndef nullptr
#define nullptr 0
#endif
#endif

class RooFIter
{
  public:
  inline RooFIter() : _ptr (0) {}
  inline RooFIter(const RooLinkedList* list) : _ptr (list->_first) {}
    
  inline RooAbsArg *next() { 
    // Return next element in collection
    if (!_ptr) return 0 ;
    TObject* arg = _ptr->_arg ;      
    _ptr = _ptr->_next;
    return (RooAbsArg*) arg ;
  }
    
 private:
    const RooLinkedListElem* _ptr ;  //! Next link element
};



class RooLinkedListIter : public TIterator {
public:

  RooLinkedListIter() {
    // coverity[UNINIT_CTOR]
  } ;


  RooLinkedListIter(const RooLinkedList* list, Bool_t forward) : 
    TIterator(), _forward(forward), _list(list)
  {
    // Constructor from list with direction
    _ptr = _list->_first ;
    _cptr = _ptr;
  }

  RooLinkedListIter(const RooLinkedListIter& other) :
    TIterator(other),
    _forward(other._forward),
    _cptr(other._cptr),
    _ptr(other._ptr), 
    _list(other._list)
  {
    // Copy constructor
  }
  
  virtual ~RooLinkedListIter() { ; }
  
  TIterator& operator=(const TIterator& other) {

    // Iterator assignment operator

    if (&other==this) return *this ;
    const RooLinkedListIter* iter = dynamic_cast<const RooLinkedListIter*>(&other) ;
    if (iter) {
      _list = iter->_list ;
      _ptr = iter->_ptr ;
      _cptr = iter->_cptr;
      _forward = iter->_forward ;
    }
    return *this ;
  }
    
  virtual const TCollection *GetCollection() const { 
    // Dummy
    return 0 ; 
  }

  virtual TObject *Next() { 
    // Return next element in collection
    if (!_ptr) return 0 ;
    _cptr = _ptr;
    TObject* arg = _ptr->_arg ;      
    _ptr = _forward ? _ptr->_next : _ptr->_prev ;
    return arg ;
  }

  TObject *NextNV() { 
    // Return next element in collection
    if (!_ptr) return 0 ;
    _cptr = _ptr;
    TObject* arg = _ptr->_arg ;      
    _ptr = _forward ? _ptr->_next : _ptr->_prev ;
    return arg ;
  }
  

  virtual void Reset() { 
    // Return iterator to first element in collection
    _ptr = _forward ? _list->_first : _list->_last ;
    _cptr = _ptr;
  }

  bool operator!=(const TIterator &aIter) const {
    if (nullptr == &aIter)
       return _cptr;
    if ((aIter.IsA() == RooLinkedListIter::Class())) {
       const RooLinkedListIter &iter(dynamic_cast<const RooLinkedListIter &>(aIter));
       return (_cptr != iter._cptr);
    }
    return false; // for base class we don't implement a comparison
  }

  bool operator!=(const RooLinkedListIter &aIter) const {
    if (nullptr == (&aIter))
      return _cptr;
     
    return (_cptr != aIter._cptr);
  }

  virtual TObject *operator*() const {
    // Return element iterator points to
    return (_cptr ? _cptr->_arg : nullptr);
  }

protected:
  Bool_t _forward ;                //  Iterator direction
  const RooLinkedListElem* _cptr ; //! Current link element
  const RooLinkedListElem* _ptr ;  //! Next link element
  const RooLinkedList* _list ;     //! Collection iterated over

  ClassDef(RooLinkedListIter,1) // Iterator for RooLinkedList container class
} ;




#endif
