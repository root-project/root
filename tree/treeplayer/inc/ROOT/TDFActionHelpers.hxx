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

#include "ROOT/TDFUtils.hxx"
#include "ROOT/TThreadedObject.hxx"
#include "TH1.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

/// \cond HIDDEN_SYMBOLS

namespace ROOT {
namespace Internal {
namespace TDF {

using Count_t = unsigned long;
using Hist_t = ::TH1F;

template <typename F>
class ForeachSlotHelper {
   F fCallable;

public:
   using BranchTypes_t = typename TRemoveFirst<typename TFunctionTraits<F>::Args_t>::Types_t;
   ForeachSlotHelper(F &&f) : fCallable(f) {}

   template <typename... Args>
   void Exec(unsigned int slot, Args &&... args)
   {
      // check that the decayed types of Args are the same as the branch types
      static_assert(std::is_same<TTypeList<typename std::decay<Args>::type...>, BranchTypes_t>::value, "");
      fCallable(slot, std::forward<Args>(args)...);
   }

   void Finalize() { /* noop */}
};

class CountHelper {
   std::shared_ptr<unsigned int> fResultCount;
   std::vector<Count_t> fCounts;

public:
   using BranchTypes_t = TTypeList<>;
   CountHelper(const std::shared_ptr<unsigned int> &resultCount, unsigned int nSlots);
   void Exec(unsigned int slot);
   void Finalize();
};

class FillHelper {
   // this sets a total initial size of 16 MB for the buffers (can increase)
   static constexpr unsigned int fgTotalBufSize = 2097152;
   using BufEl_t = double;
   using Buf_t = std::vector<BufEl_t>;

   std::vector<Buf_t> fBuffers;
   std::vector<Buf_t> fWBuffers;
   std::shared_ptr<Hist_t> fResultHist;
   unsigned int fNSlots;
   unsigned int fBufSize;
   Buf_t fMin;
   Buf_t fMax;

   void UpdateMinMax(unsigned int slot, double v);

public:
   FillHelper(const std::shared_ptr<Hist_t> &h, unsigned int nSlots);
   void Exec(unsigned int slot, double v);
   void Exec(unsigned int slot, double v, double w);

   template <typename T, typename std::enable_if<TIsContainer<T>::fgValue, int>::type = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      auto &thisBuf = fBuffers[slot];
      for (auto &v : vs) {
         UpdateMinMax(slot, v);
         thisBuf.emplace_back(v); // TODO: Can be optimised in case T == BufEl_t
      }
   }

   template <typename T, typename W,
             typename std::enable_if<TIsContainer<T>::fgValue && TIsContainer<W>::fgValue, int>::type = 0>
   void Exec(unsigned int slot, const T &vs, const W &ws)
   {
      auto &thisBuf = fBuffers[slot];
      for (auto &v : vs) {
         UpdateMinMax(slot, v);
         thisBuf.emplace_back(v); // TODO: Can be optimised in case T == BufEl_t
      }

      auto &thisWBuf = fWBuffers[slot];
      for (auto &w : ws) {
         thisWBuf.emplace_back(w); // TODO: Can be optimised in case T == BufEl_t
      }
   }

   void Finalize();
};

extern template void FillHelper::Exec(unsigned int, const std::vector<float> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<double> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<char> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<int> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<unsigned int> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<float> &, const std::vector<float> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<double> &, const std::vector<double> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<char> &, const std::vector<char> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<int> &, const std::vector<int> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<unsigned int> &,
                                      const std::vector<unsigned int> &);

template <typename HIST = Hist_t>
class FillTOHelper {
   std::unique_ptr<TThreadedObject<HIST>> fTo;

public:
   FillTOHelper(FillTOHelper &&) = default;

   FillTOHelper(const std::shared_ptr<HIST> &h, unsigned int nSlots) : fTo(new TThreadedObject<HIST>(*h))
   {
      fTo->SetAtSlot(0, h);
      // Initialise all other slots
      for (unsigned int i = 0; i < nSlots; ++i) {
         fTo->GetAtSlot(i);
      }
   }

   void Exec(unsigned int slot, double x0) // 1D histos
   {
      fTo->GetAtSlotUnchecked(slot)->Fill(x0);
   }

   void Exec(unsigned int slot, double x0, double x1) // 1D weighted and 2D histos
   {
      fTo->GetAtSlotUnchecked(slot)->Fill(x0, x1);
   }

   void Exec(unsigned int slot, double x0, double x1, double x2) // 2D weighted and 3D histos
   {
      fTo->GetAtSlotUnchecked(slot)->Fill(x0, x1, x2);
   }

   void Exec(unsigned int slot, double x0, double x1, double x2, double x3) // 3D weighted histos
   {
      fTo->GetAtSlotUnchecked(slot)->Fill(x0, x1, x2, x3);
   }

   template <typename X0, typename std::enable_if<TIsContainer<X0>::fgValue, int>::type = 0>
   void Exec(unsigned int slot, const X0 &x0s)
   {
      auto thisSlotH = fTo->GetAtSlotUnchecked(slot);
      for (auto &x0 : x0s) {
         thisSlotH->Fill(x0); // TODO: Can be optimised in case T == vector<double>
      }
   }

   template <typename X0, typename X1,
             typename std::enable_if<TIsContainer<X0>::fgValue && TIsContainer<X1>::fgValue, int>::type = 0>
   void Exec(unsigned int slot, const X0 &x0s, const X1 &x1s)
   {
      auto thisSlotH = fTo->GetAtSlotUnchecked(slot);
      if (x0s.size() != x1s.size()) {
         throw std::runtime_error("Cannot fill histogram with values in containers of different sizes.");
      }
      auto x0sIt = std::begin(x0s);
      const auto x0sEnd = std::end(x0s);
      auto x1sIt = std::begin(x1s);
      for (; x0sIt != x0sEnd; x0sIt++, x1sIt++) {
         thisSlotH->Fill(*x0sIt, *x1sIt); // TODO: Can be optimised in case T == vector<double>
      }
   }

   template <typename X0, typename X1, typename X2,
             typename std::enable_if<
                TIsContainer<X0>::fgValue && TIsContainer<X1>::fgValue && TIsContainer<X2>::fgValue, int>::type = 0>
   void Exec(unsigned int slot, const X0 &x0s, const X1 &x1s, const X2 &x2s)
   {
      auto thisSlotH = fTo->GetAtSlotUnchecked(slot);
      if (!(x0s.size() == x1s.size() && x1s.size() == x2s.size())) {
         throw std::runtime_error("Cannot fill histogram with values in containers of different sizes.");
      }
      auto x0sIt = std::begin(x0s);
      const auto x0sEnd = std::end(x0s);
      auto x1sIt = std::begin(x1s);
      auto x2sIt = std::begin(x2s);
      for (; x0sIt != x0sEnd; x0sIt++, x1sIt++, x2sIt++) {
         thisSlotH->Fill(*x0sIt, *x1sIt, *x2sIt); // TODO: Can be optimised in case T == vector<double>
      }
   }
   template <typename X0, typename X1, typename X2, typename X3,
             typename std::enable_if<TIsContainer<X0>::fgValue && TIsContainer<X1>::fgValue &&
                                        TIsContainer<X2>::fgValue && TIsContainer<X3>::fgValue,
                                     int>::type = 0>
   void Exec(unsigned int slot, const X0 &x0s, const X1 &x1s, const X2 &x2s, const X3 &x3s)
   {
      auto thisSlotH = fTo->GetAtSlotUnchecked(slot);
      if (!(x0s.size() == x1s.size() && x1s.size() == x2s.size() && x1s.size() == x3s.size())) {
         throw std::runtime_error("Cannot fill histogram with values in containers of different sizes.");
      }
      auto x0sIt = std::begin(x0s);
      const auto x0sEnd = std::end(x0s);
      auto x1sIt = std::begin(x1s);
      auto x2sIt = std::begin(x2s);
      auto x3sIt = std::begin(x3s);
      for (; x0sIt != x0sEnd; x0sIt++, x1sIt++, x2sIt++, x3sIt++) {
         thisSlotH->Fill(*x0sIt, *x1sIt, *x2sIt, *x3sIt); // TODO: Can be optimised in case T == vector<double>
      }
   }
   void Finalize() { fTo->Merge(); }
};

// note: changes to this class should probably be replicated in its partial
// specialization below
template <typename T, typename COLL>
class TakeHelper {
   std::vector<std::shared_ptr<COLL>> fColls;

public:
   using BranchTypes_t = TTypeList<T>;
   TakeHelper(const std::shared_ptr<COLL> &resultColl, unsigned int nSlots)
   {
      fColls.emplace_back(resultColl);
      for (unsigned int i = 1; i < nSlots; ++i) fColls.emplace_back(std::make_shared<COLL>());
   }

   template <typename V, typename std::enable_if<!TIsContainer<V>::fgValue, int>::type = 0>
   void Exec(unsigned int slot, V v)
   {
      fColls[slot]->emplace_back(v);
   }

   template <typename V, typename std::enable_if<TIsContainer<V>::fgValue, int>::type = 0>
   void Exec(unsigned int slot, const V &vs)
   {
      auto thisColl = fColls[slot];
      thisColl.insert(std::begin(thisColl), std::begin(vs), std::begin(vs));
   }

   void Finalize()
   {
      auto rColl = fColls[0];
      for (unsigned int i = 1; i < fColls.size(); ++i) {
         auto &coll = fColls[i];
         for (T &v : *coll) {
            rColl->emplace_back(v);
         }
      }
   }
};

// note: changes to this class should probably be replicated in its unspecialized
// declaration above
template <typename T>
class TakeHelper<T, std::vector<T>> {
   std::vector<std::shared_ptr<std::vector<T>>> fColls;

public:
   using BranchTypes_t = TTypeList<T>;
   TakeHelper(const std::shared_ptr<std::vector<T>> &resultColl, unsigned int nSlots)
   {
      fColls.emplace_back(resultColl);
      for (unsigned int i = 1; i < nSlots; ++i) {
         auto v = std::make_shared<std::vector<T>>();
         v->reserve(1024);
         fColls.emplace_back(v);
      }
   }

   template <typename V, typename std::enable_if<!TIsContainer<V>::fgValue, int>::type = 0>
   void Exec(unsigned int slot, V v)
   {
      fColls[slot]->emplace_back(v);
   }

   template <typename V, typename std::enable_if<TIsContainer<V>::fgValue, int>::type = 0>
   void Exec(unsigned int slot, const V &vs)
   {
      auto thisColl = fColls[slot];
      thisColl->insert(std::begin(thisColl), std::begin(vs), std::begin(vs));
   }

   void Finalize()
   {
      ULong64_t totSize = 0;
      for (auto &coll : fColls) totSize += coll->size();
      auto rColl = fColls[0];
      rColl->reserve(totSize);
      for (unsigned int i = 1; i < fColls.size(); ++i) {
         auto &coll = fColls[i];
         rColl->insert(rColl->end(), coll->begin(), coll->end());
      }
   }
};

template <typename F, typename T>
class ReduceHelper {
   F fReduceFun;
   std::shared_ptr<T> fReduceRes;
   std::vector<T> fReduceObjs;

public:
   using BranchTypes_t = TTypeList<T>;
   ReduceHelper(F &&f, const std::shared_ptr<T> &reduceRes, unsigned int nSlots)
      : fReduceFun(std::move(f)), fReduceRes(reduceRes), fReduceObjs(nSlots, *reduceRes)
   {
   }

   void Exec(unsigned int slot, const T &value) { fReduceObjs[slot] = fReduceFun(fReduceObjs[slot], value); }

   void Finalize()
   {
      for (auto &t : fReduceObjs) *fReduceRes = fReduceFun(*fReduceRes, t);
   }
};

class MinHelper {
   std::shared_ptr<double> fResultMin;
   std::vector<double> fMins;

public:
   MinHelper(const std::shared_ptr<double> &minVPtr, unsigned int nSlots);
   void Exec(unsigned int slot, double v);

   template <typename T, typename std::enable_if<TIsContainer<T>::fgValue, int>::type = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs) fMins[slot] = std::min((double)v, fMins[slot]);
   }

   void Finalize();
};

extern template void MinHelper::Exec(unsigned int, const std::vector<float> &);
extern template void MinHelper::Exec(unsigned int, const std::vector<double> &);
extern template void MinHelper::Exec(unsigned int, const std::vector<char> &);
extern template void MinHelper::Exec(unsigned int, const std::vector<int> &);
extern template void MinHelper::Exec(unsigned int, const std::vector<unsigned int> &);

class MaxHelper {
   std::shared_ptr<double> fResultMax;
   std::vector<double> fMaxs;

public:
   MaxHelper(const std::shared_ptr<double> &maxVPtr, unsigned int nSlots);
   void Exec(unsigned int slot, double v);

   template <typename T, typename std::enable_if<TIsContainer<T>::fgValue, int>::type = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs) fMaxs[slot] = std::max((double)v, fMaxs[slot]);
   }

   void Finalize();
};

extern template void MaxHelper::Exec(unsigned int, const std::vector<float> &);
extern template void MaxHelper::Exec(unsigned int, const std::vector<double> &);
extern template void MaxHelper::Exec(unsigned int, const std::vector<char> &);
extern template void MaxHelper::Exec(unsigned int, const std::vector<int> &);
extern template void MaxHelper::Exec(unsigned int, const std::vector<unsigned int> &);

class MeanHelper {
   std::shared_ptr<double> fResultMean;
   std::vector<Count_t> fCounts;
   std::vector<double> fSums;

public:
   MeanHelper(const std::shared_ptr<double> &meanVPtr, unsigned int nSlots);
   void Exec(unsigned int slot, double v);

   template <typename T, typename std::enable_if<TIsContainer<T>::fgValue, int>::type = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs) {
         fSums[slot] += v;
         fCounts[slot]++;
      }
   }

   void Finalize();
};

extern template void MeanHelper::Exec(unsigned int, const std::vector<float> &);
extern template void MeanHelper::Exec(unsigned int, const std::vector<double> &);
extern template void MeanHelper::Exec(unsigned int, const std::vector<char> &);
extern template void MeanHelper::Exec(unsigned int, const std::vector<int> &);
extern template void MeanHelper::Exec(unsigned int, const std::vector<unsigned int> &);

} // end of NS TDF
} // end of NS Internal
} // end of NS ROOT

/// \endcond

#endif
