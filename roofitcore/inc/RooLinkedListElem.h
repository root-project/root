/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
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
#ifndef ROO_LINKED_LIST_ELEM
#define ROO_LINKED_LIST_ELEM

#include "Rtypes.h"
#include "RooFitCore/RooAbsArg.hh"
#include "RooFitCore/RooLinkedListElem.hh"

class RooLinkedListElem ;
class TBuffer ;

class RooLinkedListElem {
public:
  // Initial element ctor
  RooLinkedListElem(RooAbsArg* arg) : 
    _arg(arg), _prev(0), _next(0){
  }

  // Link element ctor
  RooLinkedListElem(RooAbsArg* arg, RooLinkedListElem* after) : 
    _arg(arg), _prev(after), _next(after->_next) {

    // Insert self in link
    after->_next = this ;
    if (_next) _next->_prev = this ;
  }

  // Destructor
  virtual ~RooLinkedListElem() {    
    // Remove self from link
    if (_prev) _prev->_next = _next ;
    if (_next) _next->_prev = _prev ;
  }

protected:
  friend class RooLinkedList ;
  friend class RooLinkedListIter ;
  RooLinkedListElem* _prev ; // Link to previous element in list
  RooLinkedListElem* _next ; // Link to next element in list
  RooAbsArg* _arg ;          // Link to contents

protected:

  // Forbidden
  RooLinkedListElem(const RooLinkedListElem&) ;

  ClassDef(RooLinkedListElem,0) // Element of RooLinkedList container class
} ;



#endif
