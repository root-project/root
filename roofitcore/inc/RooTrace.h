/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooTrace.rdl,v 1.1 2001/08/02 21:39:13 verkerke Exp $
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
    
    memset(_pad1,0,1000) ;
    memset(_pad2,0,1000) ;
    //cout << "RooTraceObj::ctor obj=" << obj << " prev=" << _prev << " next=" << _next << endl ;
  }

  inline checkPad() {
    Int_t i ;
    for(i=0 ; i<1000 ; i++) {
      if (_pad1[i]!=0) cout << "RooTraceObj(" << _obj << ") pad1[" << i << "] = (" << (void*)(_pad1+i) << ") = " << (char) _pad1[i] << endl ;
    }

    for(i=0 ; i<1000 ; i++) {
      if (_pad2[i]!=0) cout << "RooTraceObj(" << _obj << ") pad2[" << i << "] = (" << (void*)(_pad2+1) << ") = " << (char) _pad2[i] << endl ;
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

  inline static void create(const TObject* obj) { if (_active) create2(obj) ; }
  inline static void destroy(const TObject* obj) { if (_active) destroy2(obj) ; }
  static void create2(const TObject* obj) ;
  static void destroy2(const TObject* obj) ;
  
  inline static void active(Bool_t flag) { _active = flag ; }
  inline static void verbose(Bool_t flag) { _verbose = flag ; }
  inline static void pad(Bool_t flag) { _pad = flag ; }
  

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
    RooTraceObj* link2(0) ;
    while(link) {      
      link->checkPad() ;
      if (link->obj() == obj) {
	link2 = link ;
      }
      link = link->next() ;
    }
  
    if (link2) {
      delete link2 ;
      return obj ;
    }

    return 0 ;
  }


  static void dump(ostream& os=cout) ;

  static Bool_t _active ;
  static Bool_t _verbose ;
  static Bool_t _pad ;
  static RooTraceObj* _traceList ;

protected:

  ClassDef(RooTrace,1) // Memory tracer utility for RooFitTools objects
};


#endif
