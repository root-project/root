/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNormSetCache.rdl,v 1.10 2005/06/20 15:44:55 wverkerke Exp $
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
#ifndef ROO_NORMSET_CACHE
#define ROO_NORMSET_CACHE

#include "Riostream.h"
#include <assert.h>
#include "Rtypes.h"
#include "RooNameSet.h"
#include "RooSetPair.h"
#include "RooHashTable.h"

class RooArgSet ;
class RooSetPair ;

typedef RooArgSet* pRooArgSet ;

class RooNormSetCache {

public:
  RooNormSetCache(Int_t regSize=10) ;
  RooNormSetCache(const RooNormSetCache& other) ;
  virtual ~RooNormSetCache() ;

  void add(const RooArgSet* set1, const RooArgSet* set2=0) ;

  inline Int_t index(const RooArgSet* set1, const RooArgSet* set2=0, const TNamed* set2RangeName=0) {

    // Match range name first
    if (set2RangeName != _set2RangeName) return -1 ;

    Int_t i ;
    for (i=0 ; i<_nreg ; i++) {
      if (_asArr[i]._set1 == set1 && _asArr[i]._set2 == set2) return i ;
    }
    return -1 ;
  }

  inline Bool_t contains(const RooArgSet* set1, const RooArgSet* set2=0, const TNamed* set2RangeName=0) {
    if (set2RangeName!=_set2RangeName) return kFALSE ;
    if (_htable) return (_htable->findSetPair(set1,set2)) ;
    return (index(set1,set2,set2RangeName)>=0) ;
  }

  inline Bool_t containsSet1(const RooArgSet* set1) {
    Int_t i ;
    for (i=0 ; i<_nreg ; i++) {
      if (_asArr[i]._set1 == set1) return kTRUE ;
    }
    return kFALSE ;
  }

  const RooArgSet* lastSet1() const { return _nreg>0?_asArr[_nreg-1]._set1:0 ; }
  const RooArgSet* lastSet2() const { return _nreg>0?_asArr[_nreg-1]._set2:0 ; }
  const RooNameSet& nameSet1() const { return _name1 ; }
  const RooNameSet& nameSet2() const { return _name2 ; }

  Bool_t autoCache(const RooAbsArg* self, const RooArgSet* set1, const RooArgSet* set2=0, const TNamed* set2RangeName=0, Bool_t autoRefill=kTRUE) ;
    
  void clear() ;
  Int_t entries() const { return _nreg ; }

protected:

  friend class RooNormListManager ;
  friend class RooNormManager ;
  void initialize(const RooNormSetCache& other) ;

  void expand() ;

  RooHashTable* _htable ; //! do not persist
  Int_t _regSize ;
  Int_t _nreg ;
  RooSetPair* _asArr ;  //! do not persist

  RooNameSet _name1 ;   //!
  RooNameSet _name2 ;   //!
  TNamed*    _set2RangeName ; //!

  ClassDef(RooNormSetCache,1) // Manager class for a single PDF normalization integral
} ;

#endif 
