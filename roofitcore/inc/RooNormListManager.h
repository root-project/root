/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$                                                             *
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
