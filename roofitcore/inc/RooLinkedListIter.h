/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooLinkedListIter.rdl,v 1.10 2005/04/18 21:44:48 wverkerke Exp $
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
#include "RooAbsArg.h"
#include "RooLinkedList.h"

class RooLinkedListIter : public TIterator {
public:
  RooLinkedListIter(const RooLinkedList* list, Bool_t forward) : 
    TIterator(), _forward(forward), _list(list)
  {
    _ptr = _list->_first ;
  }

  RooLinkedListIter(const RooLinkedListIter& other) :
    TIterator(other),
    _forward(other._forward),
    _ptr(other._ptr), 
    _list(other._list)
  {
  }
  
  virtual ~RooLinkedListIter() { ; }
  
  TIterator& operator=(const TIterator& other) {

    if (&other==this) return *this ;
    const RooLinkedListIter* iter = dynamic_cast<const RooLinkedListIter*>(&other) ;
    if (iter) {
      _list = iter->_list ;
      _ptr = iter->_ptr ;
      _forward = iter->_forward ;
    }
    return *this ;
  }

    
  virtual const TCollection *GetCollection() const { 
    return 0 ; 
  }

  virtual TObject *Next() { 
    if (!_ptr) return 0 ;
    TObject* arg = _ptr->_arg ;      
      _ptr = _forward ? _ptr->_next : _ptr->_prev ;
      return arg ;
  }

  virtual void Reset() { 
    _ptr = _forward ? _list->_first : _list->_last ;
  }

protected:
  Bool_t _forward ;                //  Iterator direction
  const RooLinkedListElem* _ptr ;  //! Current link element
  const RooLinkedList* _list ;     //! Collection iterated over

  ClassDef(RooLinkedListIter,0) // Iterator for RooLinkedList container class
} ;


#endif
