/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_TRACE
#define ROO_TRACE

#include <iostream.h>
#include <assert.h>
#include "TNamed.h"
#include "TList.h"

typedef TObject* pTObject ;


class RooTraceObj {
public:
  inline RooTraceObj(const TObject* obj, RooTraceObj* prev, RooTraceObj* next) {
    if (next) next->_prev = this ;
    if (prev) prev->_next = this ;
    _obj = obj ;
    _prev = prev ;
    _next = next ;
    
    Int_t i ;
    for(i=0 ; i<1000 ; i++) {
      _pad1[i]=0 ;
      _pad2[i]=0 ;
    }
    
    //cout << "RooTraceObj::ctor obj=" << obj << " prev=" << _prev << " next=" << _next << endl ;
  }

  inline checkPad() {
    Int_t i ;
    for(i=0 ; i<1000 ; i++) {
      if (_pad1[i]!=0) cout << "RooTraceObj pad1[" << i << "] = (" << (void*)(_pad1+i) << ") = " << (char) _pad1[i] << endl ;
    }

    for(i=0 ; i<1000 ; i++) {
      if (_pad2[i]!=0) cout << "RooTraceObj pad2[" << i << "] = (" << (void*)(_pad2+1) << ") = " << (char) _pad2[i] << endl ;
    }

  }

  inline ~RooTraceObj() {
    if (_next) { _next->_prev = _prev ; }
    if (_prev) { _prev->_next = _next ; }
    //cout << "RooTraceObj::dtor next->prev=" << _prev << " prev->next=" << _next << endl ;
  }

  inline RooTraceObj* next() { return _next ; }
  inline RooTraceObj* prev() { return _prev ; }
  inline const TObject* obj() { return _obj ; }

protected:  
  char _pad1[1000] ;
  const TObject* _obj ;
  RooTraceObj* _prev ;
  RooTraceObj* _next ;
  char _pad2[1000] ;
} ;


class RooTrace {
public:
  
  static void create(const TObject* obj) ;
  static void destroy(const TObject* obj) ;
  
  inline static void level(Int_t level) { 
    _traceLevel = level ; 
  }
  

  static void checkPad()  ;
  
  static void addToList(const TObject* obj) {
    if (!_traceList) {
      _traceList = new RooTraceObj(obj,0,0) ;
    } else {
      RooTraceObj* link(_traceList) ;
      while(link->next()) {
	link = link->next() ;
      }
      new RooTraceObj(obj,link,0) ;
    }
  }
  
  static const TObject* removeFromList(const TObject* obj) {
    RooTraceObj* link(_traceList) ;
    while(link) {      
      link->checkPad() ;
      if (link->obj() == obj) {
	delete link ;
	return obj ;
      }
      link = link->next() ;
    }
  
    return 0 ;
  }


  static void dump(ostream& os=cout) ;
  static Int_t _traceLevel ;
  static RooTraceObj* _traceList ;

protected:

  ClassDef(RooTrace,1) // Memory tracer utility for RooFitTools objects
};


#endif
