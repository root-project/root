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

#include <TError.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime> // for CPU time measurement with clock()
#include <memory>
#include <ostream>
#include <string>
#include <vector>

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
   std::atomic<std::int64_t> fCounter = 0;
   bool fIsActive = false;

public:
   void Activate() { fIsActive = true; }
   bool IsActive() const { return fIsActive; }

   void Inc() {
      if (fIsActive)
         ++fCounter;
   }
   void Dec() {
      if (fIsActive)
         --fCounter;
   }
   void Add(int64_t delta) {
      if (fIsActive)
         fCounter += delta;
   }
   int64_t XAdd(int64_t delta) {
      if (fIsActive)
         return fCounter.fetch_add(delta);
      return 0;
   }
   int64_t GetValue() const {
      if (fIsActive)
         return fCounter.load();
      return 0;
   }
   void SetValue(int64_t val) {
      if (fIsActive)
         fCounter.store(val);
   }

   virtual std::string ToString() const { return std::to_string(GetValue()); }
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RNTupleTickCounter
\ingroup NTuple
\brief A performance counter for CPU ticks
*/
// clang-format on

class RNTupleTickCounter : public RNTuplePerfCounter {
public:
   /// Return the result in ns
   std::string ToString() const final;
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RNTupleTimer
\ingroup NTuple
\brief Record wall time and CPU time between construction and destruction

Uses RAII as a stop watch. Only the wall time counter is used to determine whether the timer is active.
*/
// clang-format on
class RNTupleTimer {
private:
   using Clock_t = std::chrono::steady_clock;

   RNTuplePerfCounter &fCtrWallTime;
   RNTuplePerfCounter &fCtrCpuTicks;
   /// Wall clock time
   Clock_t::time_point fStartTime;
   /// CPU time
   clock_t fStartTicks;

public:
   RNTupleTimer(RNTuplePerfCounter &ctrWallTime, RNTuplePerfCounter &ctrCpuTicks)
      : fCtrWallTime(ctrWallTime), fCtrCpuTicks(ctrCpuTicks)
   {
      if (!fCtrWallTime.IsActive())
         return;
      fStartTime = Clock_t::now();
      fStartTicks = clock();
   }

   ~RNTupleTimer() {
      if (!fCtrWallTime.IsActive())
         return;
      auto wallTimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock_t::now() - fStartTime);
      fCtrWallTime.Add(wallTimeNs.count());
      fCtrCpuTicks.Add(clock() - fStartTicks);
   }

   RNTupleTimer(const RNTupleTimer &other) = delete;
   RNTupleTimer &operator =(const RNTupleTimer &other) = delete;
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RNTupleMetrics
\ingroup NTuple
\brief A collection of Counter objects with a name, a unit, and a description.

The class owns the counters; on registration of a new
*/
// clang-format on
class RNTupleMetrics {
public:
   struct RCounterInfo {
      std::string fName;
      std::string fUnit;
      std::string fDescription;
      RCounterInfo(const std::string &name, const std::string &unit, const std::string &desc)
         : fName(name), fUnit(unit), fDescription(desc) {}
      bool operator== (const RCounterInfo &other) const { return fName == other.fName; }
      bool operator== (const std::string &name) const { return fName == name; }
   };

private:
   std::vector<std::unique_ptr<RNTuplePerfCounter>> fCounters;
   std::vector<RCounterInfo> fCounterInfos;
   std::string fName;
   bool fIsActive = false;

public:
   explicit RNTupleMetrics(const std::string &name) : fName(name) {}
   RNTupleMetrics(const RNTupleMetrics &other) = delete;
   RNTupleMetrics & operator=(const RNTupleMetrics &other) = delete;
   ~RNTupleMetrics() = default;

   template <typename CounterT = RNTuplePerfCounter>
   RNTuplePerfCounter *Generate(const RCounterInfo &info) {
      R__ASSERT(Lookup(info.fName) == nullptr);
      auto &counter = fCounters.emplace_back(std::make_unique<RNTuplePerfCounter>());
      fCounterInfos.emplace_back(info);
      return counter.get();
   }
   template <typename CounterT = RNTuplePerfCounter>
   RNTuplePerfCounter *Generate(const std::string &name, const std::string &unit, const std::string &desc) {
      return Generate<CounterT>(RCounterInfo(name, unit, desc));
   }
   RNTuplePerfCounter *Lookup(const std::string &name) const;
   void Print(std::ostream &output) const;
   void Activate();
   bool IsActive() const { return fIsActive; }
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
