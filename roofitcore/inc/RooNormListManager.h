/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   25-Sep-2001 WV Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/
#ifndef ROO_NORM_LIST_MANAGER
#define ROO_NORM_LIST_MANAGER

#include "Rtypes.h"

#include "RooFitCore/RooNormSetCache.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooArgList.hh"

class RooNameSet ;
typedef RooArgList* pRooArgList ;

class RooNormListManager {

public:
  RooNormListManager(Int_t maxSize=10) ;
  RooNormListManager(const RooNormListManager& other) ;
  virtual ~RooNormListManager() ;
  
  RooArgList* getNormList(const RooAbsArg* self, const RooArgSet* nset, const RooArgSet* iset=0) ;
  Int_t setNormList(const RooAbsArg* self, const RooArgSet* nset, const RooArgSet* iset, RooArgList* normColl) ;  
  void reset() ;

  Int_t lastIndex() const { return _lastIndex ; }
  RooArgList* getNormListByIndex(Int_t index) const ;
  Int_t cacheSize() const { return _size ; }

  void setVerbose(Bool_t flag=kTRUE) const { _verbose = flag ; }
 
protected:

  Int_t _maxSize ;
  Int_t _size ;
  Int_t _lastIndex ;
  static Bool_t _verbose ;

  RooNormSetCache* _nsetCache ; //!
  pRooArgList* _normList ; //!

  ClassDef(RooNormListManager,1) // Manager class for PDF normalization integral lists
} ;

#endif 
