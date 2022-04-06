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

/**
\file RooNormSetCache.cxx
\class RooNormSetCache
\ingroup Roofitcore

Class RooNormSet cache manage the bookkeeping of multiple instances
of sets of integration and normalization observables that effectively
have the same definition. In complex function expression many
RooArgSets with the same contents may be passed to an object that
caches intermediate results dependent on the normalization/integration set
To avoid unnecessary cache faulting, This class tracks all instances
with the same contents and reports to the owner if the present nset/iset
is truely different from the current reference. Class RooNormSet only
evaluates each RooArgSet pointer once, it therefore assumes that
RooArgSets with normalization and/or integration sets are not changes
during their lifetime.
**/

#include <RooNormSetCache.h>
#include <RooHelpers.h>

////////////////////////////////////////////////////////////////////////////////
/// Clear contents

void RooNormSetCache::clear()
{
  {
    PairIdxMapType tmpmap;
    tmpmap.swap(_pairToIdx);
  }
  {
    PairVectType tmppairvect;
    tmppairvect.swap(_pairs);
  }
  _next = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Add given pair of RooArgSet pointers to our store

void RooNormSetCache::add(const RooArgSet* set1, const RooArgSet* set2)
{
  const Pair pair(set1, set2);
  PairIdxMapType::iterator it = _pairToIdx.lower_bound(pair);
  if (_pairToIdx.end() != it && it->first == pair) {
    // not empty, and keys match - nothing to do
    return;
  }
  // register pair -> index mapping
  _pairToIdx.insert(it, std::make_pair(pair, ULong_t(_pairs.size())));
  // save pair at that index
  _pairs.push_back(pair);
  // if the cache grew too large, start replacing in a round-robin fashion
  while (_pairs.size() > _max) {
    // new index of the pair: replace slot _next
    it->second = _next;
    // find and erase mapping of old pair in that slot
    _pairToIdx.erase(_pairs[_next]);
    // put new pair into new slot
    _pairs[_next] = _pairs.back();
    // and erase the copy we no longer need
    _pairs.erase(_pairs.end() - 1);
    ++_next;
    _next %= _max;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// If RooArgSets set1 and set2 or sets with similar contents have
/// been seen by this cache manager before return `false` If not,
/// return `true`. If sets have not been seen and doRefill is true,
/// update cache reference to current input sets.

Bool_t RooNormSetCache::autoCache(const RooAbsArg* self, const RooArgSet* set1,
   const RooArgSet* set2, const TNamed* set2RangeName, Bool_t doRefill)
{

  // Automated cache management function - Returns `true` if cache is invalidated

  // A - Check if set1/2 are in cache and range name is identical
  if (set2RangeName == _set2RangeName && contains(set1,set2)) {
    return false ;
  }

  // B - Check if dependents(set1/set2) are compatible with current cache

//   cout << "RooNormSetCache::autoCache set1 = " << (set1?*set1:RooArgSet()) << " set2 = " << (set2?*set2:RooArgSet()) << endl;
//   if (set1) set1->Print("v");
//   if (set2) set2->Print("v");
  //if (self) self->Print("v");

  RooArgSet set1d;
  RooArgSet set2d;
  if (self) {
    if(set1) self->getObservables(set1,set1d,false);
    if(set2) self->getObservables(set2,set2d,false);
  } else {
    if(set1) set1->snapshot(set1d);
    if(set2) set2->snapshot(set2d);
  }

//   cout << "RooNormSetCache::autoCache set1d = " << *set1d << " set2 = " << *set2d << endl;

  using RooHelpers::getColonSeparatedNameString;

  if (   getColonSeparatedNameString(set1d) == _name1
      && getColonSeparatedNameString(set2d) == _name2
      && _set2RangeName == set2RangeName) {
    // Compatible - Add current set1/2 to cache
    add(set1,set2);

    return false;
  }

  // C - Reset cache and refill with current state
  if (doRefill) {
    clear();
    add(set1,set2);
    _name1 = getColonSeparatedNameString(set1d);
    _name2 = getColonSeparatedNameString(set2d);
//     cout << "RooNormSetCache::autoCache() _name1 refilled from " << *set1d << " to " ; _name1.printValue(cout) ; cout << endl;
//     cout << "RooNormSetCache::autoCache() _name2 refilled from " << *set2d << " to " ; _name2.printValue(cout) ; cout << endl;
    _set2RangeName = (TNamed*) set2RangeName;
  }

  return true;
}
