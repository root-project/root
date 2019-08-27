/// \file ROOT/RNTupleMetrics.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-08-27
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleMetrics
#define ROOT7_RNTupleMetrics

#include <atomic>
#include <cstdint>
#include <string>

namespace ROOT {
namespace Experimental {
namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::Detail::RNTuplePerfCounter
\ingroup NTuple
\brief A wrapper around an atomic counter

A counter can keep track, e.g., of the number of operations, their length, or a number of bytes read/transferred/...
*/
// clang-format on
class RNTuplePerfCounter {
private:
   std::atomic<std::int64_t> fCounter;

public:
   void Inc() { ++fCounter; }
   void Dec() { --fCounter; }
   void Add(int64_t delta) { fCounter += delta; }
   int64_t XAdd(int64_t delta) { return fCounter.fetch_add(delta); }
   int64_t GetValue() const { return fCounter.load(); }
   void SetValue(int64_t val) { fCounter.store(val); }

   std::string ToString() const;
   std::string ToStringK() const;
   std::string ToStringKi() const;
   std::string ToStringM() const;
   std::string ToStringMi() const;
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
