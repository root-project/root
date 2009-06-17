/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// Class RooNormSet cache manage the bookkeeping of multiple instances
// of sets of integration and normalization observables that effectively
// have the same definition. In complex function expression many
// RooArgSets with the same contents may be passed to an object that
// caches intermediate results dependent on the normalization/integration set
// To avoid unnecessary cache faulting, This class tracks all instances
// with the same contents and reports to the owner if the present nset/iset
// is truely different from the current reference. Class RooNormSet only
// evaluates each RooArgSet pointer once, it therefore assumes that
// RooArgSets with normalization and/or integration sets are not changes
// during their lifetime. 
// END_HTML
//
#include "RooFit.h"

#include "RooNormSetCache.h"
#include "RooNormSetCache.h"
#include "RooArgSet.h"

ClassImp(RooNormSetCache)
;


#include <iostream>
using namespace std ;

//_____________________________________________________________________________
RooNormSetCache::RooNormSetCache(Int_t regSize) :
  _htable(0), _regSize(regSize), _nreg(0), _asArr(0), _set2RangeName(0)
{
  // Construct normalization set manager with given initial size
  _htable = regSize>16 ? new RooHashTable(regSize,RooHashTable::Intrinsic) : 0 ;
}



//_____________________________________________________________________________
RooNormSetCache::RooNormSetCache(const RooNormSetCache& other) :
  _htable(0), _regSize(other._regSize), _nreg(0), _asArr(0), _set2RangeName(0)
{
  // Copy constructor

  _htable = _regSize>16 ? new RooHashTable(_regSize,RooHashTable::Intrinsic) : 0 ;
}



//_____________________________________________________________________________
RooNormSetCache::~RooNormSetCache() 
{
  // Destructor

  delete[] _asArr ;
  if (_htable) delete _htable ;
}



//_____________________________________________________________________________
void RooNormSetCache::clear()
{
  // Clear contents 
  _nreg = 0 ;  
  if (_htable) {
    delete _htable ;
    _htable = 0 ;
  }
}



//_____________________________________________________________________________
void RooNormSetCache::initialize(const RooNormSetCache& other) 
{
  // Initialize cache from contents of given other cache
  clear() ;

  Int_t i ;
  for (i=0 ; i<other._nreg ; i++) {
    add(other._asArr[i]._set1,other._asArr[i]._set2) ;
  }

  _name1 = other._name1 ;
  _name2 = other._name2 ;

  _set2RangeName = other._set2RangeName ;
}



//_____________________________________________________________________________
void RooNormSetCache::add(const RooArgSet* set1, const RooArgSet* set2)
{
  // Add given pair of RooArgSet pointers to our store

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


//_____________________________________________________________________________
void RooNormSetCache::expand()
{
  // Expand registry size by doubling capacity

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



//_____________________________________________________________________________
Bool_t RooNormSetCache::autoCache(const RooAbsArg* self, const RooArgSet* set1, const RooArgSet* set2, const TNamed* set2RangeName, Bool_t doRefill) 
{
  // If RooArgSets set1 and set2 or sets with similar contents have
  // been seen by this cache manager before return kFALSE If not,
  // return kTRUE. If sets have not been seen and doRefill is true,
  // update cache reference to current input sets.
  

  // Automated cache management function - Returns kTRUE if cache is invalidated
  
  // A - Check if set1/2 are in cache and range name is identical
  if (set2RangeName==_set2RangeName && contains(set1,set2)) {
    return kFALSE ;
  }

  // B - Check if dependents(set1/set2) are compatible with current cache
  RooNameSet nset1d,nset2d ;

//   cout << "RooNormSetCache::autoCache set1 = " << (set1?*set1:RooArgSet()) << " set2 = " << (set2?*set2:RooArgSet()) << endl ;
//   if (set1) set1->Print("v") ;
//   if (set2) set2->Print("v") ;
  //if (self) self->Print("v") ;

  RooArgSet *set1d, *set2d ;
  if (self) {
    set1d = set1 ? self->getObservables(*set1,kFALSE) : new RooArgSet ;
    set2d = set2 ? self->getObservables(*set2,kFALSE) : new RooArgSet ;
  } else {
    set1d = set1 ? (RooArgSet*)set1->snapshot() : new RooArgSet ;
    set2d = set2 ? (RooArgSet*)set2->snapshot() : new RooArgSet ;
  }

//   cout << "RooNormSetCache::autoCache set1d = " << *set1d << " set2 = " << *set2d << endl ;

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
//     cout << "RooNormSetCache::autoCache() _name1 refilled from " << *set1d << " to " ; _name1.printValue(cout) ; cout << endl ;
//     cout << "RooNormSetCache::autoCache() _name2 refilled from " << *set2d << " to " ; _name2.printValue(cout) ; cout << endl ;
    _set2RangeName = (TNamed*) set2RangeName ;
  }
  
  delete set1d ;
  delete set2d ;
  return kTRUE ;
}
