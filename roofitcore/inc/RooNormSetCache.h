/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   25-Sep-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_NORMSET_CACHE
#define ROO_NORMSET_CACHE

#include <iostream.h>
#include <assert.h>
#include "Rtypes.h"
#include "RooFitCore/RooNameSet.hh"

class RooArgSet ;

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
      if (_asArr1[i] == set1 && _asArr2[i] == set2) return i ;
    }
    return -1 ;
  }

  inline Bool_t contains(const RooArgSet* set1, const RooArgSet* set2=0) {
    return (index(set1,set2)>=0) ;
  }

  const RooArgSet* lastSet1() const { return _nreg>0?_asArr1[_nreg-1]:0 ; }
  const RooArgSet* lastSet2() const { return _nreg>0?_asArr2[_nreg-1]:0 ; }
  const RooNameSet& nameSet1() const { return _name1 ; }
  const RooNameSet& nameSet2() const { return _name2 ; }

  Bool_t autoCache(const RooAbsArg* self, const RooArgSet* set1, const RooArgSet* set2=0, Bool_t autoRefill=kTRUE) ;
  
  
  void clear() ;
  Int_t entries() const { return _nreg ; }

protected:

  friend class RooNormListManager ;
  friend class RooNormManager ;
  void initialize(const RooNormSetCache& other) ;

  Int_t _regSize ;
  Int_t _nreg ;
  pRooArgSet* _asArr1;  //! do not persist
  pRooArgSet* _asArr2;  //! do not persist
  RooNameSet _name1 ;   //!
  RooNameSet _name2 ;   //!

  ClassDef(RooNormSetCache,1) // Manager class for a single PDF normalization integral
} ;

#endif 
