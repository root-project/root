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
\file RooAICRegistry.cxx
\class RooAICRegistry
\ingroup Roofitcore

Utility class for operator p.d.f
classes that keeps track of analytical integration codes and
associated normalization and integration sets.
**/

#include <RooAICRegistry.h>
#include <RooArgSet.h>

namespace {

template<class RooArgSetPointer_t>
std::unique_ptr<RooArgSet> makeSnapshot(RooArgSetPointer_t const &set)
{
   if (!set)
      return nullptr;
   auto out = std::make_unique<RooArgSet>();
   set->snapshot(*out, false);
   return out;
}

} // namespace

RooAICRegistry::RooAICRegistry(std::size_t size)
{
   _clArr.reserve(size);
   for (auto &a : _asArr) {
      a.reserve(size);
   }
}

RooAICRegistry::RooAICRegistry(const RooAICRegistry &other) : _clArr(other._clArr)
{
   // Copy code-list array if other PDF has one
   for (std::size_t iArr = 0; iArr < _asArr.size(); ++iArr) {
      _asArr[iArr].resize(_clArr.size());
      for (std::size_t i = 0; i < _clArr.size(); ++i) {
         _asArr[iArr][i] = makeSnapshot(other._asArr[iArr][i]);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Copy assignment.

RooAICRegistry &RooAICRegistry::operator=(const RooAICRegistry &other)
{
   RooAICRegistry tmp(other);
   *this = std::move(tmp);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Store given arrays of integer codes, and up to four RooArgSets in
/// the registry (each setX pointer may be null). The registry
/// clones all RooArgSets internally so the RooArgSets passed as
/// arguments do not need to live beyond the store() call. The return
/// value is a unique master code for the given configuration of
/// integers and RooArgSets. If an identical combination is
/// previously stored in the registry no objects are stored and the
/// unique code of the existing entry is returned.

int RooAICRegistry::store(const std::vector<Int_t> &codeList, RooArgSet *set1, RooArgSet *set2, RooArgSet *set3,
                            RooArgSet *set4)
{
   // Taking the ownership of the input sets
   std::array<RooArgSet *, 4> sets{set1, set2, set3, set4};

   // Loop over code-list array
   for (std::size_t i = 0; i < _clArr.size(); ++i) {
      // Existing slot, compare with current list, if matched return index.
      // First., check that array contents is identical.
      bool match = _clArr[i] == codeList;

      // Check that supplied configuration of lists is identical
      for (std::size_t iArr = 0; iArr < 4; ++iArr) {
         if ((_asArr[iArr][i] && !sets[iArr]) || (!_asArr[iArr][i] && sets[iArr]))
            match = false;
      }

      // Check that contents of arrays is identical
      for (std::size_t iArr = 0; iArr < 4; ++iArr) {
         if (_asArr[iArr][i] && sets[iArr] && !sets[iArr]->equals(*_asArr[iArr][i]))
            match = false;
      }

      if (match) {
         return i;
      }
   }

   // Store code list and return index
   _clArr.push_back(codeList);
   for (std::size_t iArr = 0; iArr < 4; ++iArr) {
      _asArr[iArr].emplace_back(makeSnapshot(sets[iArr]));
   }

   return _clArr.size() - 1;
}
