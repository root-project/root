// Author: Enrico Guiraud, Danilo Piparo CERN  12/2016

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDFOPERATIONS
#define ROOT_TDFOPERATIONS

#include "ROOT/TDFTraitsUtils.hxx"
#include "ROOT/TThreadedObject.hxx"

#include <algorithm>
#include <memory>
#include <limits>
#include <vector>

/// \cond HIDDEN_SYMBOLS

class TH1F;

namespace ROOT {

namespace Internal {

namespace Operations {

using namespace Internal::TDFTraitsUtils;
using Count_t = unsigned long;

class CountOperation {
   unsigned int *fResultCount;
   std::vector<Count_t> fCounts;

public:
   CountOperation(unsigned int *resultCount, unsigned int nSlots) : fResultCount(resultCount), fCounts(nSlots, 0) {}

   void Exec(unsigned int slot)
   {
      fCounts[slot]++;
   }

   ~CountOperation()
   {
      *fResultCount = 0;
      for (auto &c : fCounts) {
         *fResultCount += c;
      }
   }
};

template<typename HIST>
class FillOperation {
   // this sets a total initial size of 16 MB for the buffers (can increase)
   static constexpr unsigned int fgTotalBufSize = 2097152;
   using BufEl_t = double;
   using Buf_t = std::vector<BufEl_t>;

   std::vector<Buf_t> fBuffers;
   std::shared_ptr<HIST> fResultHist;
   unsigned int fBufSize;
   Buf_t fMin;
   Buf_t fMax;

   template <typename T>
   void UpdateMinMax(unsigned int slot, T v) {
      auto& thisMin = fMin[slot];
      auto& thisMax = fMax[slot];
      thisMin = std::min(thisMin, (BufEl_t)v);
      thisMax = std::max(thisMax, (BufEl_t)v);
   }

public:
   FillOperation(std::shared_ptr<HIST> h, unsigned int nSlots) : fResultHist(h),
                                                                 fBufSize (fgTotalBufSize / nSlots),
                                                                 fMin(nSlots, std::numeric_limits<BufEl_t>::max()),
                                                                 fMax(nSlots, std::numeric_limits<BufEl_t>::min())
   {
      fBuffers.reserve(nSlots);
      for (unsigned int i=0; i<nSlots; ++i) {
         Buf_t v;
         v.reserve(fBufSize);
         fBuffers.emplace_back(v);
      }
   }

   template <typename T, typename std::enable_if<!TIsContainer<T>::fgValue, int>::type = 0>
   void Exec(T v, unsigned int slot)
   {
      UpdateMinMax(slot, v);
      fBuffers[slot].emplace_back(v);
   }

   template <typename T, typename std::enable_if<TIsContainer<T>::fgValue, int>::type = 0>
   void Exec(const T &vs, unsigned int slot)
   {
      auto& thisBuf = fBuffers[slot];
      for (auto& v : vs) {
         UpdateMinMax(slot, v);
         thisBuf.emplace_back(v); // TODO: Can be optimised in case T == BufEl_t
      }
   }

   ~FillOperation()
   {

      BufEl_t globalMin = *std::min_element(fMin.begin(), fMin.end());
      BufEl_t globalMax = *std::max_element(fMax.begin(), fMax.end());

      if (fResultHist->CanExtendAllAxes() &&
          globalMin != std::numeric_limits<BufEl_t>::max() &&
          globalMax != std::numeric_limits<BufEl_t>::min()) {
         auto xaxis = fResultHist->GetXaxis();
         fResultHist->ExtendAxis(globalMin, xaxis);
         fResultHist->ExtendAxis(globalMax, xaxis);
      }

      for (auto& buf : fBuffers) {
         Buf_t w(buf.size(),1); // A bug in FillN?
         fResultHist->FillN(buf.size(), buf.data(),  w.data());
      }
   }
};

template<typename HIST>
class FillTOOperation {
   TThreadedObject<HIST> fTo;

public:

   FillTOOperation(std::shared_ptr<HIST> h, unsigned int nSlots) : fTo(*h)
   {
      fTo.SetAtSlot(0, h);
      // Initialise all other slots
      for (unsigned int i = 0 ; i < nSlots; ++i) {
         fTo.GetAtSlot(i);
      }
   }

   template <typename T, typename std::enable_if<!TIsContainer<T>::fgValue, int>::type = 0>
   void Exec(T v, unsigned int slot)
   {
      fTo.GetAtSlotUnchecked(slot)->Fill(v);
   }

   template <typename T, typename std::enable_if<TIsContainer<T>::fgValue, int>::type = 0>
   void Exec(const T &vs, unsigned int slot)
   {
      auto thisSlotH = fTo.GetAtSlotUnchecked(slot);
      for (auto& v : vs) {
         thisSlotH->Fill(v); // TODO: Can be optimised in case T == vector<double>
      }
   }

   ~FillTOOperation()
   {
      fTo.Merge();
   }

};

// note: changes to this class should probably be replicated in its partial
// specialization below
template<typename T, typename COLL>
class TakeOperation {
   std::vector<std::shared_ptr<COLL>> fColls;
public:
   TakeOperation(std::shared_ptr<COLL> resultColl, unsigned int nSlots)
   {
      fColls.emplace_back(resultColl);
      for (unsigned int i = 1; i < nSlots; ++i)
         fColls.emplace_back(std::make_shared<COLL>());
   }

   template <typename V, typename std::enable_if<!TIsContainer<V>::fgValue, int>::type = 0>
   void Exec(V v, unsigned int slot)
   {
      fColls[slot]->emplace_back(v);
   }

   template <typename V, typename std::enable_if<TIsContainer<V>::fgValue, int>::type = 0>
   void Exec(const V &vs, unsigned int slot)
   {
      auto thisColl = fColls[slot];
      thisColl.insert(std::begin(thisColl), std::begin(vs), std::begin(vs));
   }

   ~TakeOperation()
   {
      auto rColl = fColls[0];
      for (unsigned int i = 1; i < fColls.size(); ++i) {
         auto& coll = fColls[i];
         for (T &v : *coll) {
            rColl->emplace_back(v);
         }
      }
   }
};

// note: changes to this class should probably be replicated in its unspecialized
// declaration above
template<typename T>
class TakeOperation<T, std::vector<T>> {
   std::vector<std::shared_ptr<std::vector<T>>> fColls;
public:
   TakeOperation(std::shared_ptr<std::vector<T>> resultColl, unsigned int nSlots)
   {
      fColls.emplace_back(resultColl);
      for (unsigned int i = 1; i < nSlots; ++i) {
         auto v = std::make_shared<std::vector<T>>();
         v->reserve(1024);
         fColls.emplace_back(v);
      }
   }

   template <typename V, typename std::enable_if<!TIsContainer<V>::fgValue, int>::type = 0>
   void Exec(V v, unsigned int slot)
   {
      fColls[slot]->emplace_back(v);
   }

   template <typename V, typename std::enable_if<TIsContainer<V>::fgValue, int>::type = 0>
   void Exec(const V &vs, unsigned int slot)
   {
      auto thisColl = fColls[slot];
      thisColl->insert(std::begin(thisColl), std::begin(vs), std::begin(vs));
   }

   ~TakeOperation()
   {
      unsigned int totSize = 0;
      for (auto& coll : fColls) totSize += coll->size();
      auto rColl = fColls[0];
      rColl->reserve(totSize);
      for (unsigned int i = 1; i < fColls.size(); ++i) {
         auto& coll = fColls[i];
         rColl->insert(rColl->end(), coll->begin(), coll->end());
      }
   }
};

class MinOperation {
   double *fResultMin;
   std::vector<double> fMins;

public:
   MinOperation(double *minVPtr, unsigned int nSlots)
      : fResultMin(minVPtr), fMins(nSlots, std::numeric_limits<double>::max()) { }
   template <typename T, typename std::enable_if<!TIsContainer<T>::fgValue, int>::type = 0>
   void Exec(T v, unsigned int slot)
   {
      fMins[slot] = std::min((double)v, fMins[slot]);
   }
   template <typename T, typename std::enable_if<TIsContainer<T>::fgValue, int>::type = 0>
   void Exec(const T &vs, unsigned int slot)
   {
      for (auto &&v : vs) fMins[slot] = std::min((double)v, fMins[slot]);
   }
   ~MinOperation()
   {
      *fResultMin = std::numeric_limits<double>::max();
      for (auto &m : fMins) *fResultMin = std::min(m, *fResultMin);
   }
};

class MaxOperation {
   double *fResultMax;
   std::vector<double> fMaxs;

public:
   MaxOperation(double *maxVPtr, unsigned int nSlots)
      : fResultMax(maxVPtr), fMaxs(nSlots, std::numeric_limits<double>::min()) { }
   template <typename T, typename std::enable_if<!TIsContainer<T>::fgValue, int>::type = 0>
   void Exec(T v, unsigned int slot)
   {
      fMaxs[slot] = std::max((double)v, fMaxs[slot]);
   }

   template <typename T, typename std::enable_if<TIsContainer<T>::fgValue, int>::type = 0>
   void Exec(const T &vs, unsigned int slot)
   {
      for (auto &&v : vs) fMaxs[slot] = std::max((double)v, fMaxs[slot]);
   }

   ~MaxOperation()
   {
      *fResultMax = std::numeric_limits<double>::min();
      for (auto &m : fMaxs) {
         *fResultMax = std::max(m, *fResultMax);
      }
   }
};

class MeanOperation {
   double *fResultMean;
   std::vector<Count_t> fCounts;
   std::vector<double> fSums;

public:
   MeanOperation(double *meanVPtr, unsigned int nSlots) : fResultMean(meanVPtr), fCounts(nSlots, 0), fSums(nSlots, 0) {}
   template <typename T, typename std::enable_if<!TIsContainer<T>::fgValue, int>::type = 0>
   void Exec(T v, unsigned int slot)
   {
      fSums[slot] += v;
      fCounts[slot] ++;
   }

   template <typename T, typename std::enable_if<TIsContainer<T>::fgValue, int>::type = 0>
   void Exec(const T &vs, unsigned int slot)
   {
      for (auto &&v : vs) {
         fSums[slot] += v;
         fCounts[slot]++;
      }
   }

   ~MeanOperation()
   {
      double sumOfSums = 0;
      for (auto &s : fSums) sumOfSums += s;
      Count_t sumOfCounts = 0;
      for (auto &c : fCounts) sumOfCounts += c;
      *fResultMean = sumOfSums / (sumOfCounts > 0 ? sumOfCounts : 1);
   }
};

} // end of NS Operations

} // end of NS Internal

} // end of NS ROOT

/// \endcond

#endif
