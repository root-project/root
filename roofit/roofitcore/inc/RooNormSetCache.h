/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNormSetCache.h,v 1.12 2007/08/09 19:55:47 wouter Exp $
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

#include <RooArgSet.h>

#include <Rtypes.h>

#include <vector>
#include <map>
#include <string>

typedef RooArgSet* pRooArgSet ;

class RooNormSetCache {
protected:
  class Pair {
  public:
    using Value_t = RooFit::UniqueId<RooArgSet>::Value_t;
    Pair(const RooArgSet* set1, const RooArgSet* set2)
      : _pair{RooFit::getUniqueId(set1), RooFit::getUniqueId(set2)} {}
   bool operator==(Pair const &other) const { return _pair == other._pair; }
   bool operator<(Pair const &other) const { return _pair < other._pair; }

   Value_t const& first() const { return _pair.first; }
   Value_t const& second() const { return _pair.second; }
  private:
    std::pair<Value_t,Value_t> _pair;
  };

  typedef std::vector<Pair> PairVectType;
  typedef std::map<Pair, ULong64_t> PairIdxMapType;

public:
  RooNormSetCache(ULong_t max = 32);
  virtual ~RooNormSetCache();

  void add(const RooArgSet* set1, const RooArgSet* set2 = 0);

  inline Int_t index(const RooArgSet* set1, const RooArgSet* set2 = 0,
      const TNamed* set2RangeName = 0)
  {
    // Match range name first
    if (set2RangeName != _set2RangeName) return -1;
    const Pair pair(set1, set2);
    PairIdxMapType::const_iterator it = _pairToIdx.lower_bound(pair);
    if (_pairToIdx.end() != it && it->first == pair) {
      return it->second;
    }
    return -1;
  }

  inline Bool_t contains(const RooArgSet* set1, const RooArgSet* set2 = 0,
      const TNamed* set2RangeName = 0)
  { return (index(set1,set2,set2RangeName) >= 0); }

  inline Bool_t containsSet1(const RooArgSet* set1)
  {
    const Pair pair(set1, (const RooArgSet*)0);
    PairIdxMapType::const_iterator it = _pairToIdx.lower_bound(pair);
    if (_pairToIdx.end() != it && it->first.first() == RooFit::getUniqueId(set1))
      return kTRUE;
    return kFALSE;
  }

  const std::string& nameSet1() const { return _name1; }
  const std::string& nameSet2() const { return _name2; }

  Bool_t autoCache(const RooAbsArg* self, const RooArgSet* set1,
      const RooArgSet* set2 = 0, const TNamed* set2RangeName = 0,
      Bool_t autoRefill = kTRUE);

  void clear();
  Int_t entries() const { return _pairs.size(); }

  void initialize(const RooNormSetCache& other) { clear(); *this = other; }

protected:

  PairVectType _pairs; ///<!
  PairIdxMapType _pairToIdx; ///<!
  ULong_t _max; ///<!
  ULong_t _next; ///<!

  std::string _name1;   ///<!
  std::string _name2;   ///<!
  TNamed*    _set2RangeName; ///<!

  ClassDef(RooNormSetCache, 0) // Management tool for tracking sets of similar integration/normalization sets
};

#endif
