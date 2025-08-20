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

#include <deque>
#include <set>
#include <string>

class RooNormSetCache {

public:
  RooNormSetCache(std::size_t max = 32) : _max(max) {}

  inline bool contains(const RooArgSet* set1, const RooArgSet* set2 = nullptr,
      const TNamed* set2RangeName = nullptr)
  {
    // Match range name first
    if (set2RangeName != _set2RangeName) return false;
    return _pairSet.find({RooFit::getUniqueId(set1), RooFit::getUniqueId(set2)}) != _pairSet.end();
  }

  const std::string& nameSet1() const { return _name1; }
  const std::string& nameSet2() const { return _name2; }

  bool autoCache(const RooAbsArg* self, const RooArgSet* set1,
      const RooArgSet* set2 = nullptr, const TNamed* set2RangeName = nullptr,
      bool autoRefill = true);

  void clear();

private:

  void add(const RooArgSet* set1, const RooArgSet* set2 = nullptr);

  using Value_t = RooFit::UniqueId<RooArgSet>::Value_t;
  using Pair_t = std::pair<Value_t,Value_t>;

  std::deque<Pair_t> _pairs; ///<!
  std::set<Pair_t> _pairSet; ///<!
  std::size_t _max; ///<!

  std::string _name1;   ///<!
  std::string _name2;   ///<!
  TNamed*    _set2RangeName = nullptr; ///<!
};

#endif
