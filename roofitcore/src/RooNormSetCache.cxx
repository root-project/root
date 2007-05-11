/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNormSetCache.cc,v 1.14 2005/06/16 09:31:29 wverkerke Exp $
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

// -- CLASS DESCRIPTION [AUX] --
#include "RooFit.h"

#include "RooNormSetCache.h"
#include "RooNormSetCache.h"
#include "RooArgSet.h"

ClassImp(RooNormSetCache)
;

RooNormSetCache::RooNormSetCache(Int_t regSize) :
  _htable(0), _regSize(regSize), _nreg(0), _asArr(0), _set2RangeName(0)
{
  _htable = regSize>16 ? new RooHashTable(regSize,RooHashTable::Intrinsic) : 0 ;
}



RooNormSetCache::RooNormSetCache(const RooNormSetCache& other) :
  _htable(0), _regSize(other._regSize), _nreg(0), _asArr(0), _set2RangeName(0)
{
  _htable = _regSize>16 ? new RooHashTable(_regSize,RooHashTable::Intrinsic) : 0 ;
}



RooNormSetCache::~RooNormSetCache() 
{
  delete[] _asArr ;
  if (_htable) delete _htable ;
}



void RooNormSetCache::clear()
{
  _nreg = 0 ;  
  if (_htable) {
    delete _htable ;
    _htable = 0 ;
  }
}


void RooNormSetCache::initialize(const RooNormSetCache& other) 
{
  clear() ;

  Int_t i ;
  for (i=0 ; i<other._nreg ; i++) {
    add(other._asArr[i]._set1,other._asArr[i]._set2) ;
  }

  _name1 = other._name1 ;
  _name2 = other._name2 ;
  _set2RangeName = other._set2RangeName ;
}



void RooNormSetCache::add(const RooArgSet* set1, const RooArgSet* set2)
{
  // If code list array has never been used, allocate and initialize here
  if (!_asArr) {
    _asArr = new RooSetPair[_regSize] ;
  }

  if (!contains(set1,set2)) {
    // Add to cache
    _asArr[_nreg]._set1 = (RooArgSet*)set1 ;
    _asArr[_nreg]._set2 = (RooArgSet*)set2 ;
    if (_htable) _htable->add((TObject*)&_asArr[_nreg]) ;
    _nreg++ ;
  }

  // Expand cache if full 
  if (_nreg==_regSize) expand() ;

}

void RooNormSetCache::expand()
{
  Int_t newSize = _regSize*2 ;

  if (_htable) {
    delete _htable ;
    _htable = 0 ;
  }

  // Allocate increased buffer 
  RooSetPair* asArr_new = new RooSetPair[newSize] ;
  if (newSize>16) {
    //cout << "RooNormSetCache::add() instantiating hash table with size " << newSize << endl ;
    _htable = new RooHashTable(newSize,RooHashTable::Intrinsic) ;
  }

  // Copy old buffer 
  Int_t i ;
  for (i=0 ; i<_nreg ; i++) {
    asArr_new[i]._set1 = _asArr[i]._set1 ;
    asArr_new[i]._set2 = _asArr[i]._set2 ;
    if (_htable) _htable->add((TObject*)&asArr_new[i]) ;
  }
  
  // Delete old buffers 
  delete[] _asArr ;

  // Install new buffers
  _asArr = asArr_new ;
  _regSize = newSize ;
}


Bool_t RooNormSetCache::autoCache(const RooAbsArg* self, const RooArgSet* set1, const RooArgSet* set2, const TNamed* set2RangeName, Bool_t doRefill) 
{
  // Automated cache management function - Returns kTRUE if cache is invalidated
  
  // A - Check if set1/2 are in cache and range name is identical
  if (set2RangeName==_set2RangeName && contains(set1,set2)) {
    return kFALSE ;
  }

  // B - Check if dependents(set1/set2) are compatible with current cache
  RooNameSet nset1d,nset2d ;

  RooArgSet* set1d = set1 ? self->getObservables(*set1) : new RooArgSet ;
  RooArgSet* set2d = set2 ? self->getObservables(*set2) : new RooArgSet ;

  nset1d.refill(*set1d) ;
  nset2d.refill(*set2d) ;

  if (nset1d==_name1&&nset2d==_name2&&_set2RangeName==set2RangeName) {
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
    _set2RangeName = (TNamed*) set2RangeName ;
  }
  
  delete set1d ;
  delete set2d ;
  return kTRUE ;
}
