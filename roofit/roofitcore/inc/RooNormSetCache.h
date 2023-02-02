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

#include <vector>
#include <map>
#include <string>

class RooNormSetCache {
private:
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
  typedef std::map<Pair, std::size_t> PairIdxMapType;

public:
  RooNormSetCache(std::size_t max = 32) : _max(max) {}

  void add(const RooArgSet* set1, const RooArgSet* set2 = nullptr);

  inline int index(const RooArgSet* set1, const RooArgSet* set2 = nullptr,
      const TNamed* set2RangeName = nullptr)
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

  inline bool contains(const RooArgSet* set1, const RooArgSet* set2 = nullptr,
      const TNamed* set2RangeName = nullptr)
  { return (index(set1,set2,set2RangeName) >= 0); }

  inline bool containsSet1(const RooArgSet* set1)
  {
    const Pair pair(set1, (const RooArgSet*)0);
    PairIdxMapType::const_iterator it = _pairToIdx.lower_bound(pair);
    if (_pairToIdx.end() != it && it->first.first() == (RooNormSetCache::Pair::Value_t)RooFit::getUniqueId(set1))
      return true;
    return false;
  }

  const std::string& nameSet1() const { return _name1; }
  const std::string& nameSet2() const { return _name2; }

  bool autoCache(const RooAbsArg* self, const RooArgSet* set1,
      const RooArgSet* set2 = nullptr, const TNamed* set2RangeName = nullptr,
      bool autoRefill = true);

  void clear();
  std::size_t entries() const { return _pairs.size(); }

  void initialize(const RooNormSetCache& other) { clear(); *this = other; }

private:

  PairVectType _pairs; ///<!
  PairIdxMapType _pairToIdx; ///<!
  std::size_t _max; ///<!
  std::size_t _next = 0; ///<!

  std::string _name1;   ///<!
  std::string _name2;   ///<!
  TNamed*    _set2RangeName = nullptr; ///<!
};

#endif
