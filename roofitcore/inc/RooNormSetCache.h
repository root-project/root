/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNormSetCache.rdl,v 1.5 2004/04/05 22:44:12 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_NORMSET_CACHE
#define ROO_NORMSET_CACHE

#include <iostream.h>
#include <assert.h>
#include "Rtypes.h"
#include "RooFitCore/RooNameSet.hh"
#include "RooFitCore/RooSetPair.hh"
#include "RooFitCore/RooHashTable.hh"

class RooArgSet ;
class RooSetPair ;

typedef RooArgSet* pRooArgSet ;

class RooNormSetCache {

public:
  RooNormSetCache(Int_t regSize=10) ;
  RooNormSetCache(const RooNormSetCache& other) ;
  virtual ~RooNormSetCache() ;

  void add(const RooArgSet* set1, const RooArgSet* set2=0) ;

  inline Int_t index(const RooArgSet* set1, const RooArgSet* set2=0) {
    Int_t i ;
    for (i=0 ; i<_nreg ; i++) {
      if (_asArr[i]._set1 == set1 && _asArr[i]._set2 == set2) return i ;
    }
    return -1 ;
  }

  inline Bool_t contains(const RooArgSet* set1, const RooArgSet* set2=0) {
    if (_htable) return (_htable->findSetPair(set1,set2)) ;
    return (index(set1,set2)>=0) ;
  }

  const RooArgSet* lastSet1() const { return _nreg>0?_asArr[_nreg-1]._set1:0 ; }
  const RooArgSet* lastSet2() const { return _nreg>0?_asArr[_nreg-1]._set2:0 ; }
  const RooNameSet& nameSet1() const { return _name1 ; }
  const RooNameSet& nameSet2() const { return _name2 ; }

  Bool_t autoCache(const RooAbsArg* self, const RooArgSet* set1, const RooArgSet* set2=0, Bool_t autoRefill=kTRUE) ;
  
  
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

  ClassDef(RooNormSetCache,1) // Manager class for a single PDF normalization integral
} ;

#endif 
