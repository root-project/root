/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNormManager.rdl,v 1.3 2002/09/05 04:33:46 verkerke Exp $
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
#ifndef ROO_NORM_MANAGER
#define ROO_NORM_MANAGER

#include "Rtypes.h"

#include "RooFitCore/RooNormSetCache.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooArgSet.hh"

class RooNameSet ;
typedef RooArgSet* pRooArgSet ;
typedef RooAbsReal* pRooAbsReal ;

class RooNormManager {

public:
  RooNormManager(Int_t maxSize=10) ;
  RooNormManager(const RooNormManager& other) ;
  virtual ~RooNormManager() ;
  
  RooAbsReal* getNormalization(const RooAbsArg* self, const RooArgSet* nset, const RooArgSet* iset=0) ;
  void setNormalization(const RooAbsArg* self, const RooArgSet* nset, const RooArgSet* iset, RooAbsReal* norm) ;  

  inline RooAbsReal* lastNorm() const { return _lastNorm ; } 
  inline RooArgSet* lastNormSet() const { return _lastNormSet ; } 
  inline RooNameSet& lastNameSet() const { return *_lastNameSet ; } 

  RooAbsArg* getNormByIndex(Int_t index) const ;
  Int_t cacheSize() const { return _size ; }

protected:

  Int_t _maxSize ;
  Int_t _size ;

  RooAbsReal* _lastNorm ;
  RooArgSet* _lastNormSet ;
  RooNameSet* _lastNameSet ;

  RooNormSetCache* _nsetCache ; //!
  pRooAbsReal* _norm ; //!

  ClassDef(RooNormManager,1) // Manager class for PDF normalization integrals
} ;

#endif 
