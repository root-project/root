/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   16-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_LINKED_LIST_ITER
#define ROO_LINKED_LIST_ITER

#include "Rtypes.h"
#include "TIterator.h"
#include "RooFitCore/RooAbsArg.hh"
#include "RooFitCore/RooLinkedList.hh"

class RooLinkedListIter : public TIterator {
public:
  RooLinkedListIter(const RooLinkedList* list, Bool_t forward) : 
    _list(list), _forward(forward) {
    _ptr = _list->_first ;
  }

  RooLinkedListIter(const RooLinkedListIter& other) :
    _list(other._list), _forward(other._forward),
    _ptr(other._ptr) {
  }
  
  virtual ~RooLinkedListIter() { ; }
  
  RooLinkedListIter& operator=(const TIterator& other) {

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
    RooAbsArg* arg = _ptr->_arg ;      
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

  ClassDef(RooLinkedListIter,0)
} ;


#endif
