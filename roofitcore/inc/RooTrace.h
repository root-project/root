/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooTrace.rdl,v 1.5 2001/08/10 22:22:54 verkerke Exp $
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

// Padding block object
class RooPad {
public:
  enum dummy { size=1024 } ;

  inline RooPad() { memset(_pad,_fil,size) ; }

  Bool_t check() {
    Bool_t ret(kFALSE) ;
    for (Int_t i=0 ; i<size ; i++) {
      if (_pad[i] != _fil) {
	cout << "RooPad::check: memory address " << (void*)(&_pad[i]) << " overwritten" << endl ;
	ret=kTRUE ;
      }
    }
    return ret ;
  }
  
protected:
  static char _fil ;
  char _pad[size] ;

  ClassDef(RooPad,0) 
} ;

typedef RooPad* pRooPad ;


// Table of padding blocks 
class RooPadTable {
public: 
  RooPadTable() ;
  void addPad(const TObject* ref, Bool_t doPad) ;
  Bool_t removePad(const TObject* ref) ;
   
  void checkPads() ;
  
protected:
  friend class RooTrace ;
  enum dummy { size=100000 } ;
  Int_t _hwm ; // High water mark
  Int_t _lfm ; // Lowest free mark
  pRooPad _padA[size] ;  
  pTObject _refA[size] ;

  ClassDef(RooPadTable,0)
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
  
  
  static void dump(ostream& os=cout) ;

  static Bool_t _active ;
  static Bool_t _verbose ;
  static Bool_t _pad ;
  
  static RooPadTable _rpt ;

protected:

  ClassDef(RooTrace,0) // Memory tracer utility for RooFitTools objects
};


#endif
