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

#include <ROOT/RConfig.hxx>
#include <ROOT/RStringView.hxx>

#include <TError.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime> // for CPU time measurement with clock()
#include <functional>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <vector>
#include <utility>
#include <map>

namespace ROOT {
namespace Experimental {
namespace Detail {

class RNTupleMetrics;

// clang-format off
/**
\class ROOT::Experimental::Detail::RNTuplePerfCounter
\ingroup NTuple
\brief A performance counter with a name and a unit, which can be activated on demand

Derived classes decide on the counter type and implement printing of the value.
*/
// clang-format on
class RNTuplePerfCounter {
private:
   /// Symbol to split name, unit, description, and value when printing
   static constexpr char kFieldSeperator = '|';

   std::string fName;
   std::string fUnit;
   std::string fDescription;
   bool fIsEnabled = false;

public:
   RNTuplePerfCounter(const std::string &name, const std::string &unit, const std::string &desc)
      : fName(name), fUnit(unit), fDescription(desc) {}
   virtual ~RNTuplePerfCounter();
   void Enable() { fIsEnabled = true; }
   bool IsEnabled() const { return fIsEnabled; }
   std::string GetName() const { return fName; }
   std::string GetDescription() const { return fDescription; }
   std::string GetUnit() const { return fUnit; }

   virtual std::int64_t GetValueAsInt() const = 0;
   virtual std::string GetValueAsString() const = 0;
   std::string ToString() const;
};


class RNTupleHistogram {
public:
   /// Interval type
   typedef std::pair<std::pair<uint64_t,uint64_t>,uint64_t> HistoInterval;

   std::string fName;
   std::string fUnit;
   std::string fDescription;

   RNTupleHistogram(const std::string &name, const std::string &unit, const std::string &desc) 
      : fName(name), fUnit(unit), fDescription(desc) {}
      
   std::string ToString() const {
      auto all = GetAll();
      std::string res = "";

      for(auto &elem : all) {
         auto l = elem.first.first;
         auto u = elem.first.second;
         auto c = elem.second;
         res += std::to_string(l) + "," + std::to_string(u) + "," + std::to_string(c) + ";";
      }

      return res;
   }

   std::string GetName() const { return fName; }
   std::string GetDescription() const { return fDescription; }
   std::string GetUnit() const { return fUnit; }

   virtual void Fill(const uint64_t &n) = 0;
   virtual uint64_t GetBinContent(const uint64_t &n) = 0;
   virtual std::vector<RNTupleHistogram::HistoInterval> GetAll() const = 0;
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RNTuplePlainCounter
\ingroup NTuple
\brief A non thread-safe integral performance counter
*/
// clang-format on
class RNTuplePlainCounter : public RNTuplePerfCounter {
private:
   std::int64_t fCounter = 0;

public:
   RNTuplePlainCounter(const std::string &name, const std::string &unit, const std::string &desc)
      : RNTuplePerfCounter(name, unit, desc)
   {
   }

   R__ALWAYS_INLINE void Inc() { ++fCounter; }
   R__ALWAYS_INLINE void Dec() { --fCounter; }
   R__ALWAYS_INLINE void Add(int64_t delta) { fCounter += delta; }
   R__ALWAYS_INLINE int64_t GetValue() const { return fCounter; }
   R__ALWAYS_INLINE void SetValue(int64_t val) { fCounter = val; }
   std::int64_t GetValueAsInt() const override { return fCounter; }
   std::string GetValueAsString() const override { return std::to_string(fCounter); }
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RNTupleAtomicCounter
\ingroup NTuple
\brief A thread-safe integral performance counter
*/
// clang-format on
class RNTupleAtomicCounter : public RNTuplePerfCounter {
private:
   std::atomic<std::int64_t> fCounter{0};

public:
   RNTupleAtomicCounter(const std::string &name, const std::string &unit, const std::string &desc)
      : RNTuplePerfCounter(name, unit, desc) { }

   R__ALWAYS_INLINE
   void Inc() {
      if (R__unlikely(IsEnabled()))
         ++fCounter;
   }

   R__ALWAYS_INLINE
   void Dec() {
      if (R__unlikely(IsEnabled()))
         --fCounter;
   }

   R__ALWAYS_INLINE
   void Add(int64_t delta) {
      if (R__unlikely(IsEnabled()))
         fCounter += delta;
   }

   R__ALWAYS_INLINE
   int64_t XAdd(int64_t delta) {
      if (R__unlikely(IsEnabled()))
         return fCounter.fetch_add(delta);
      return 0;
   }

   R__ALWAYS_INLINE
   int64_t GetValue() const {
      if (R__unlikely(IsEnabled()))
         return fCounter.load();
      return 0;
   }

   R__ALWAYS_INLINE
   void SetValue(int64_t val) {
      if (R__unlikely(IsEnabled()))
         fCounter.store(val);
   }

   std::int64_t GetValueAsInt() const override { return GetValue(); }
   std::string GetValueAsString() const override { return std::to_string(GetValue()); }
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RNTupleCalcPerf
\ingroup NTuple
\brief A metric element that computes its floating point value from other counters.
*/
// clang-format on
class RNTupleCalcPerf : public RNTuplePerfCounter {
public:
   using MetricFunc_t = std::function<std::pair<bool, double>(const RNTupleMetrics &)>;

private:
   RNTupleMetrics &fMetrics;
   const MetricFunc_t fFunc;

public:
   RNTupleCalcPerf(const std::string &name, const std::string &unit, const std::string &desc,
                   RNTupleMetrics &metrics, MetricFunc_t &&func)
      : RNTuplePerfCounter(name, unit, desc), fMetrics(metrics), fFunc(std::move(func))
   {
   }

   double GetValue() const {
      auto ret = fFunc(fMetrics);
      if (ret.first)
         return ret.second;
      return std::numeric_limits<double>::quiet_NaN();
   }

   std::int64_t GetValueAsInt() const override {
      return static_cast<std::int64_t>(GetValue());
   }

   std::string GetValueAsString() const override {
      return std::to_string(GetValue());
   }
};

// clang-format off
/**
\class ROOT::Experimental::Detail::RNTupleTickCounter
\ingroup NTuple
\brief An either thread-safe or non thread safe counter for CPU ticks

On print, the value is converted from ticks to ns.
*/
// clang-format on
template <typename BaseCounterT>
class RNTupleTickCounter : public BaseCounterT {
public:
   RNTupleTickCounter(const std::string &name, const std::string &unit, const std::string &desc)
      : BaseCounterT(name, unit, desc)
   {
      R__ASSERT(unit == "ns");
   }

   std::int64_t GetValueAsInt() const final {
      auto ticks = BaseCounterT::GetValue();
      return std::uint64_t((double(ticks) / double(CLOCKS_PER_SEC)) * (1000. * 1000. * 1000.));
   }

   std::string GetValueAsString() const final {
      return std::to_string(GetValueAsInt());
   }
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RNTupleTimer
\ingroup NTuple
\brief Record wall time and CPU time between construction and destruction

Uses RAII as a stop watch. Only the wall time counter is used to determine whether the timer is active.
*/
// clang-format on
template <typename WallTimeT, typename CpuTimeT>
class RNTupleTimer {
private:
   using Clock_t = std::chrono::steady_clock;

   WallTimeT &fCtrWallTime;
   CpuTimeT &fCtrCpuTicks;
   /// Wall clock time
   Clock_t::time_point fStartTime;
   /// CPU time
   clock_t fStartTicks = 0;

public:
   RNTupleTimer(WallTimeT &ctrWallTime, CpuTimeT &ctrCpuTicks)
      : fCtrWallTime(ctrWallTime), fCtrCpuTicks(ctrCpuTicks)
   {
      if (!fCtrWallTime.IsEnabled())
         return;
      fStartTime = Clock_t::now();
      fStartTicks = clock();
   }

   ~RNTupleTimer() {
      if (!fCtrWallTime.IsEnabled())
         return;
      auto wallTimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock_t::now() - fStartTime);
      fCtrWallTime.Add(wallTimeNs.count());
      fCtrCpuTicks.Add(clock() - fStartTicks);
   }

   RNTupleTimer(const RNTupleTimer &other) = delete;
   RNTupleTimer &operator =(const RNTupleTimer &other) = delete;
};

using RNTuplePlainTimer = RNTupleTimer<RNTuplePlainCounter, RNTupleTickCounter<RNTuplePlainCounter>>;
using RNTupleAtomicTimer = RNTupleTimer<RNTupleAtomicCounter, RNTupleTickCounter<RNTupleAtomicCounter>>;

// clang-format off
/**
\class ROOT::Experimental::Detail::RNTupleMetrics
\ingroup NTuple
\brief A collection of Counter objects with a name, a unit, and a description.

The class owns the counters; on registration of a new
*/
// clang-format on
class RNTupleMetrics {
private:
   /// Symbol to split metrics name from counter / sub metrics name
   static constexpr char kNamespaceSeperator = '.';

   std::vector<std::unique_ptr<RNTuplePerfCounter>> fCounters;
   std::vector<RNTupleMetrics *> fObservedMetrics;
   std::string fName;
   bool fIsEnabled = false;

   bool Contains(const std::string &name) const;

public:
   explicit RNTupleMetrics(const std::string &name) : fName(name) {}
   RNTupleMetrics(const RNTupleMetrics &other) = delete;
   RNTupleMetrics & operator=(const RNTupleMetrics &other) = delete;
   RNTupleMetrics(RNTupleMetrics &&other) = default;
   RNTupleMetrics & operator=(RNTupleMetrics &&other) = default;
   ~RNTupleMetrics() = default;

   // TODO(jblomer): return a reference
   template <typename CounterPtrT, class... Args>
   CounterPtrT MakeCounter(const std::string &name, Args&&... args)
   {
      R__ASSERT(!Contains(name));
      auto counter = std::make_unique<std::remove_pointer_t<CounterPtrT>>(name, std::forward<Args>(args)...);
      auto ptrCounter = counter.get();
      fCounters.emplace_back(std::move(counter));
      return ptrCounter;
   }
  
   /// Searches counters registered in this object only. Returns nullptr if `name` is not found.
   const RNTuplePerfCounter *GetLocalCounter(std::string_view name) const;
   /// Searches this object and all the observed sub metrics. `name` must start with the prefix used
   /// by this RNTupleMetrics instance. Returns nullptr if `name` is not found.
   const RNTuplePerfCounter *GetCounter(std::string_view name) const;

   void ObserveMetrics(RNTupleMetrics &observee);

   void Print(std::ostream &output, const std::string &prefix = "") const;
   void Enable();
   bool IsEnabled() const { return fIsEnabled; }
};


class RNTupleHistoInterval : RNTupleHistogram{
private:
   typedef std::pair<uint64_t, uint64_t> IntervalEntry;
   typedef std::pair<uint64_t, std::pair<uint64_t, std::atomic<std::uint64_t>*>> MappingEntry;

   std::vector<RNTupleHistoInterval::MappingEntry> bins;

   static bool compareIntervalEntries(IntervalEntry e1, IntervalEntry e2) {
      return e1.first < e2.first;
   };

   std::atomic<std::uint64_t>* GetMatchingCounter(const uint64_t &n) const {
      int64_t l = 0, r = bins.size() - 1, m;
      
      while(l <= r) {
         m = l + (r - l) / 2;

         if(bins[m].first <= n
            && n <= bins[m].second.first ) {
            return bins[m].second.second;
         }

         if(bins[m].first < n) {
            l = m + 1;
         }
         else {
            r = m - 1;
         }
      }
      
      return nullptr;
   };
public:
   RNTupleHistoInterval(
      const std::string &name, const std::string &unit, const std::string &desc,
      std::vector<std::pair<uint64_t, uint64_t>> intervals
   ) : RNTupleHistogram(name, unit, desc)
   {
      std::atomic<uint64_t> *counter = nullptr;
      // sort the intervals according to lower bound
      std::sort(intervals.begin(), intervals.end(), compareIntervalEntries);

      for(uint64_t i = 0; i < intervals.size(); i++) {
         auto elem = intervals[i];
         counter = new std::atomic<uint64_t>{0};
         auto auxPair = std::make_pair(elem.second, counter);
         bins.push_back(std::make_pair(elem.first, auxPair));
      }
   };

   void Fill(const uint64_t &n) override {
      auto counter = GetMatchingCounter(n);

      if(counter != nullptr) {
         ++(*counter);
      }
   };

   uint64_t GetBinContent(const uint64_t &n) override {
      auto counter = GetMatchingCounter(n);
      return counter == nullptr ? 0 : counter->load();
   };

   std::vector<RNTupleHistogram::HistoInterval> GetAll() const override {
      uint64_t lowBound, upperBound, count;
      std::vector<RNTupleHistogram::HistoInterval> all;

      for(auto &elem : bins) {
         lowBound = elem.first;
         upperBound = elem.second.first;
         count = elem.second.second->load();
         auto intervalPair = std::make_pair(lowBound, upperBound);
         all.push_back(std::make_pair(intervalPair, count));
      }

      return all;
   }
};

class RNTupleHistoCounterLog : RNTupleHistogram {
private:
   std::atomic<uint64_t>* slots[66];
   uint64_t fUpperBound;
   uint fBitUpperBound;

   uint ulog2(uint64_t n) const {
      uint64_t targetLevel = 0;
      
      while (n >>= 1) {
         ++targetLevel;
      }

      return targetLevel;
   };

   uint64_t pow2(uint exp) const {
      uint64_t base = 1;
      return base << exp;
   }
public:
   RNTupleHistoCounterLog(
      const std::string &name, const std::string &unit, const std::string &desc,
      const uint64_t upperBound
   ) : RNTupleHistogram(name, unit, desc), fUpperBound(upperBound), fBitUpperBound(ulog2(upperBound))
   {
      for(uint64_t i = 0; i <= fBitUpperBound + 1; i++) {
         slots[i] = new std::atomic<uint64_t>{0};
      }
   };

   uint MaxLogUpperBound() {
      return fBitUpperBound;
   }

   void Fill(const uint64_t &n) override {
      if(n > fUpperBound) {
         ++(*slots[fBitUpperBound + 1]);
      }
      else {
         auto binIdx = ulog2(n);
         ++(*slots[binIdx]);
      }
   }

   uint64_t GetBinContent(const uint64_t &n) override {
      uint64_t cnt = 0;

      if(n > fUpperBound) {
         cnt = GetOverflowCount();
      }
      else {
         cnt = GetExponentCount(ulog2(n));
      }

      return cnt;
   }

   uint64_t GetExponentCount(const uint &idx) const {
      uint64_t cnt = 0;

      if(idx <= fBitUpperBound) {
         cnt = slots[idx]->load();
      }

      return cnt;
   }

   uint64_t GetOverflowCount() const {
      return slots[fBitUpperBound + 1]->load();
   }

   uint64_t GetTotalCount() {
      uint64_t cnt = 0;

      for(uint i = 0; i <= fBitUpperBound + 1; i++) {
         cnt += slots[i]->load();
      }

      return cnt;
   }

   std::vector<RNTupleHistogram::HistoInterval> GetAll() const override {
      uint64_t lowB, upperB;
      std::vector<RNTupleHistogram::HistoInterval> all;

      for(uint i = 0; i <= fBitUpperBound; i++) {
         lowB = pow2(i);
         upperB = pow2(i + 1) - 1;
         auto interval = std::make_pair(lowB, upperB);
         all.push_back(std::make_pair(interval, slots[i]->load()));
      }

      return all;
   }
};

class RNTupleHistoActiveLearn : RNTupleHistogram {
private:
   uint64_t fNumBins;
   uint fLearningPhaseSize;
   
   std::vector<uint64_t> learningSamples;
   uint64_t minSample = UINT64_MAX;
   uint64_t maxSample = 0;
   uint64_t widthSample = 0;
   bool hasFlushed = false;

   std::vector<std::atomic<uint64_t>*> bins;
   std::atomic<uint64_t> underflow{0};
   std::atomic<uint64_t> overflow{0};

   void Flush() {
      if(!hasFlushed && learningSamples.size() >= fLearningPhaseSize) {
         widthSample = (maxSample - minSample + 1) / fNumBins;
         hasFlushed = true;

         for(auto &elem: learningSamples) {
            Fill(elem);
         }
      }
   }

   uint64_t GetMatchingIdx(const uint64_t &n) const {
      auto idx = (n - minSample) / widthSample;

      if(idx >= fNumBins) {
         idx = fNumBins - 1;
      }

      return idx;
   }
public:
   RNTupleHistoActiveLearn(
      const std::string &name, const std::string &unit, const std::string &desc,
      const uint64_t &numBins, const uint &learningPhaseSize
   ) : RNTupleHistogram(name, unit, desc), fNumBins(numBins), fLearningPhaseSize(learningPhaseSize)
   {
      bins.resize(fNumBins);

      for(uint i = 0; i < fNumBins; i++) {
         bins[i] = new std::atomic<uint64_t>{0};
      }
   };

   void Fill(const uint64_t &n) override {
      Flush();

      if(!hasFlushed) {
         if(n < minSample) {
            minSample = n;
         }
         
         if(n > maxSample) {
            maxSample = n;
         }

         learningSamples.push_back(n);
      }
      else {
         if(n < minSample) {
            ++underflow;
         }
         else if(n > maxSample) {
            ++overflow;
         }
         else {
            auto idx = GetMatchingIdx(n);

            ++(*bins[idx]);
         }
      }
   }

   bool IsFlushed() {
      return hasFlushed;
   }

   uint64_t GetMin() {
      return minSample;
   }

   uint64_t GetMax() {
      return maxSample;
   }

   uint64_t GetUnderflow() {
      return underflow.load();
   }

   uint64_t GetOverflow() {
      return overflow.load();
   }

   uint64_t GetBinContent(const uint64_t &n) override {
      uint64_t cnt = 0;

      if(n < minSample) {
         cnt = underflow.load();
      }
      else {
         if(n > maxSample) {
            cnt = overflow.load();
         }
         else {
            auto idx = GetMatchingIdx(n);
            cnt = bins[idx]->load();
         }
      }

      return cnt;
   };

   std::vector<RNTupleHistogram::HistoInterval> GetAll() const override {
      uint64_t lowB, upperB;
      std::vector<RNTupleHistogram::HistoInterval> all;

      // add underflow if necessary
      if(minSample > 0) {
         lowB = 0;
         upperB = minSample - 1;
         auto interval = std::make_pair(lowB, upperB);
         all.push_back(std::make_pair(interval, underflow.load()));
      }

      // add leading bins
      for(uint i = 0; i < fNumBins; i++) {
         lowB = minSample + widthSample*i;
         upperB = lowB + widthSample - 1;

         if(i == fNumBins - 1) {
            upperB = maxSample;
         }

         auto interval = std::make_pair(lowB, upperB);
         all.push_back(std::make_pair(interval, bins[i]->load()));
      }

      // add overflow
      lowB = maxSample + 1;
      upperB = UINT64_MAX;
      auto interval = std::make_pair(lowB, upperB);
      all.push_back(std::make_pair(interval, overflow.load()));      

      return all;
   }
};

class RNTupleFixedWidthHistogram : RNTupleHistogram {
private:
   uint64_t fWidth;
   uint64_t fOffset;
   uint64_t fBinsOffsetd;
   
   std::unordered_map<uint64_t, std::atomic<uint64_t>*> bins;


   uint64_t GetMatchingKey(const uint64_t &n) const {
      uint64_t idx = 0;

      if(n < fOffset) {
         idx = fBinsOffsetd - (fOffset - n) / fWidth;
      }
      else {
         idx = (n - fOffset) / fWidth + fBinsOffsetd + 1;
      }

      printf("%ld => key is %ld:::binsize=%ld\n" , n, idx, fBinsOffsetd);
      return idx;
   }

   bool ExistsKey(const int64_t &key) const {
      return bins.find(key) != bins.end();
   }
public:
   RNTupleFixedWidthHistogram(
      const std::string &name, const std::string &unit, const std::string &desc,
      const uint64_t &width, const uint &offset
   ) : RNTupleHistogram(name, unit, desc), fWidth(width), fOffset(offset), fBinsOffsetd(offset / width + 1)
   {
      
   }

   uint64_t GetWidth() {
      return fWidth;
   }

   uint64_t GetOffset() {
      return fOffset;
   }

   void Fill(const uint64_t &n) override {
      auto key = GetMatchingKey(n);

      if(!ExistsKey(key)) {
         bins[key] = new std::atomic<uint64_t>{1};
      }
      else {
         ++(*bins[key]);
      }
   }

   uint64_t GetBinContent(const uint64_t &n) override {
      auto key = GetMatchingKey(n);
      uint64_t cnt = 0;

      if(ExistsKey(key)) {
         cnt = bins[key]->load();
      }

      return cnt;
   }

   std::vector<RNTupleHistogram::HistoInterval> GetAll() const override {
      std::vector<RNTupleHistogram::HistoInterval> all;
      std::map<uint64_t, std::atomic<uint64_t>*> ordered(bins.begin(), bins.end());
      uint64_t lowB, upperB;
      
      for(auto it = ordered.begin(); it != ordered.end(); ++it) {
         lowB = fOffset + (uint64_t)(it->first)*fWidth;
         upperB = fOffset + ((uint64_t)(it->first)+1)*fWidth - 1;
      

         auto interval = std::make_pair(lowB, upperB);
         all.push_back(std::make_pair(interval, it->second->load()));
      }

      return all;
   }
};


} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
