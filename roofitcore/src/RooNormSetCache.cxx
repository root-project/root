/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNormSetCache.cc,v 1.6 2003/05/14 02:58:40 wverkerke Exp $
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

// -- CLASS DESCRIPTION [AUX] --
#include "RooFitCore/RooNormSetCache.hh"
#include "RooFitCore/RooArgSet.hh"

ClassImp(RooNormSetCache)
;

RooNormSetCache::RooNormSetCache(Int_t regSize) :
  _regSize(regSize), _nreg(0), _asArr1(0), _asArr2(0)
{
}


RooNormSetCache::RooNormSetCache(const RooNormSetCache& other) :
   _regSize(other._regSize), _nreg(0), _asArr1(0), _asArr2(0)
{
}



RooNormSetCache::~RooNormSetCache() 
{
  delete[] _asArr1 ;
  delete[] _asArr2 ;
}



void RooNormSetCache::clear()
{
  _nreg = 0 ;
}


void RooNormSetCache::initialize(const RooNormSetCache& other) 
{
  clear() ;

  Int_t i ;
  for (i=0 ; i<other._nreg ; i++) {
    add(other._asArr1[i],other._asArr2[i]) ;
  }

  _name1 = other._name1 ;
  _name2 = other._name2 ;
}



void RooNormSetCache::add(const RooArgSet* set1, const RooArgSet* set2)
{
  // If code list array has never been used, allocate and initialize here
  if (!_asArr1) {
    _asArr1 = new pRooArgSet[_regSize] ;
    _asArr2 = new pRooArgSet[_regSize] ;
  }

  if (!contains(set1,set2) && _nreg<_regSize) {
    // Add to cache
    _asArr1[_nreg] = (RooArgSet*) set1 ;
    _asArr2[_nreg] = (RooArgSet*) set2 ;
    _nreg++ ;
  }

}


Bool_t RooNormSetCache::autoCache(const RooAbsArg* self, const RooArgSet* set1, const RooArgSet* set2, Bool_t doRefill) 
{
  // Automated cache management function - Returns kTRUE if cache is invalidated
  
  // A - Check if set1/2 are in cache
  if (contains(set1,set2)) {
    return kFALSE ;
  }

  // B - Check if dependents(set1/set2) are compatible with current cache
  RooNameSet nset1d,nset2d ;

  RooArgSet* set1d = set1 ? self->getDependents(*set1) : new RooArgSet ;
  RooArgSet* set2d = set2 ? self->getDependents(*set2) : new RooArgSet ;

  nset1d.refill(*set1d) ;
  nset2d.refill(*set2d) ;

  if (nset1d==_name1&&nset2d==_name2) {
    // Compatible - Add current set1/2 to cache
    add(set1,set2) ;

    delete set1d ;
    delete set2d ;
    return kFALSE ;
  }
  
  // C - Reset cache and refill with current state
  if (doRefill) {
    clear() ;
    add(set1,set2) ;
    _name1.refill(*set1d) ;
    _name2.refill(*set2d) ;
  }
  
  delete set1d ;
  delete set2d ;
  return kTRUE ;
}
