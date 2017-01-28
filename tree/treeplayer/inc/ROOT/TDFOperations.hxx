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
#include <limits>
#include <memory>
#include <vector>

#include "TH1F.h"

/// \cond HIDDEN_SYMBOLS

namespace ROOT {

namespace Internal {

namespace Operations {

using namespace Internal::TDFTraitsUtils;
using Count_t = unsigned long;
using Hist_t = ::TH1F;

class CountOperation {
   unsigned int *fResultCount;
   std::vector<Count_t> fCounts;

public:
   CountOperation(unsigned int *resultCount, unsigned int nSlots);
   void Exec(unsigned int slot);
   void Finalize();
   ~CountOperation();
};

class FillOperation {
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
   FillOperation(std::shared_ptr<Hist_t> h, unsigned int nSlots);
   void Exec(double v, unsigned int slot);
   void Exec(double v, double w, unsigned int slot);

   template <typename T, typename std::enable_if<TIsContainer<T>::fgValue, int>::type = 0>
   void Exec(const T &vs, unsigned int slot)
   {
      auto& thisBuf = fBuffers[slot];
      for (auto& v : vs) {
         UpdateMinMax(slot, v);
         thisBuf.emplace_back(v); // TODO: Can be optimised in case T == BufEl_t
      }
   }

   template <typename T, typename W, typename std::enable_if<TIsContainer<T>::fgValue && TIsContainer<W>::fgValue, int>::type = 0>
   void Exec(const T &vs, const W &ws, unsigned int slot)
   {
      auto& thisBuf = fBuffers[slot];
      for (auto& v : vs) {
         UpdateMinMax(slot, v);
         thisBuf.emplace_back(v); // TODO: Can be optimised in case T == BufEl_t
      }

      auto& thisWBuf = fWBuffers[slot];
      for (auto& w : ws) {
         thisWBuf.emplace_back(w); // TODO: Can be optimised in case T == BufEl_t
      }
   }

   void Finalize();
   ~FillOperation();
};

extern template void FillOperation::Exec(const std::vector<float>&, unsigned int);
extern template void FillOperation::Exec(const std::vector<double>&, unsigned int);
extern template void FillOperation::Exec(const std::vector<char>&, unsigned int);
extern template void FillOperation::Exec(const std::vector<int>&, unsigned int);
extern template void FillOperation::Exec(const std::vector<unsigned int>&, unsigned int);
extern template void FillOperation::Exec(const std::vector<float>&, const std::vector<float>&, unsigned int);
extern template void FillOperation::Exec(const std::vector<double>&, const std::vector<double>&, unsigned int);
extern template void FillOperation::Exec(const std::vector<char>&, const std::vector<char>&, unsigned int);
extern template void FillOperation::Exec(const std::vector<int>&, const std::vector<int>&, unsigned int);
extern template void FillOperation::Exec(const std::vector<unsigned int>&, const std::vector<unsigned int>&, unsigned int);

template<typename HIST=Hist_t>
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
   void Exec(double x0, unsigned int slot) // 1D histos
   {
      fTo.GetAtSlotUnchecked(slot)->Fill(x0);
   }
   void Exec(double x0, double x1, unsigned int slot) // 1D weighted and 2D histos
   {
      fTo.GetAtSlotUnchecked(slot)->Fill(x0, x1);
   }
   void Exec(double x0, double x1, double x2, unsigned int slot) // 2D weighted and 3D histos
   {
      fTo.GetAtSlotUnchecked(slot)->Fill(x0, x1, x2);
   }
   void Exec(double x0, double x1, double x2, double x3, unsigned int slot) // 3D weighted histos
   {
      fTo.GetAtSlotUnchecked(slot)->Fill(x0, x1, x2, x3);
   }
   template <typename X0, typename std::enable_if<TIsContainer<X0>::fgValue, int>::type = 0>
   void Exec(const X0 &x0s, unsigned int slot)
   {
      auto thisSlotH = fTo.GetAtSlotUnchecked(slot);
      for (auto& x0 : x0s) {
         thisSlotH->Fill(x0); // TODO: Can be optimised in case T == vector<double>
      }
   }

   template <typename X0, typename X1,
             typename std::enable_if<TIsContainer<X0>::fgValue && TIsContainer<X1>::fgValue, int>::type = 0>
   void Exec(const X0 &x0s, const X1 &x1s, unsigned int slot)
   {
      auto thisSlotH = fTo.GetAtSlotUnchecked(slot);
      if (x0s.size() != x1s.size()) {
         throw std::runtime_error("Cannot fill histogram with values in containers of different sizes.");
      }
      auto x0sIt = std::begin(x0s);
      const auto x0sEnd = std::end(x0s);
      auto x1sIt = std::begin(x1s);
      for (;x0sIt!=x0sEnd; x0sIt++, x1sIt++) {
         thisSlotH->Fill(*x0sIt, *x1sIt); // TODO: Can be optimised in case T == vector<double>
      }
   }


   template <typename X0, typename X1, typename X2,
             typename std::enable_if<TIsContainer<X0>::fgValue && TIsContainer<X1>::fgValue && TIsContainer<X2>::fgValue, int>::type = 0>
   void Exec(const X0 &x0s, const X1 &x1s, const X2 &x2s, unsigned int slot)
   {
      auto thisSlotH = fTo.GetAtSlotUnchecked(slot);
      if (!(x0s.size() == x1s.size() == x2s.size())) {
         throw std::runtime_error("Cannot fill histogram with values in containers of different sizes.");
      }
      auto x0sIt = std::begin(x0s);
      const auto x0sEnd = std::end(x0s);
      auto x1sIt = std::begin(x1s);
      auto x2sIt = std::begin(x2s);
      for (;x0sIt!=x0sEnd; x0sIt++, x1sIt++, x2sIt++) {
         thisSlotH->Fill(*x0sIt, *x1sIt, *x2sIt); // TODO: Can be optimised in case T == vector<double>
      }
   }
   template <typename X0, typename X1, typename X2, typename X3,
             typename std::enable_if<TIsContainer<X0>::fgValue && TIsContainer<X1>::fgValue && TIsContainer<X2>::fgValue && TIsContainer<X2>::fgValue, int>::type = 0>
   void Exec(const X0 &x0s, const X1 &x1s, const X2 &x2s, const X3 &x3s, unsigned int slot)
   {
      auto thisSlotH = fTo.GetAtSlotUnchecked(slot);
      if (!(x0s.size() == x1s.size() == x2s.size() == x3s.size())) {
         throw std::runtime_error("Cannot fill histogram with values in containers of different sizes.");
      }
      auto x0sIt = std::begin(x0s);
      const auto x0sEnd = std::end(x0s);
      auto x1sIt = std::begin(x1s);
      auto x2sIt = std::begin(x2s);
      auto x3sIt = std::begin(x3s);
      for (;x0sIt!=x0sEnd; x0sIt++, x1sIt++, x2sIt++, x3sIt++) {
         thisSlotH->Fill(*x0sIt, *x1sIt, *x2sIt, *x3sIt); // TODO: Can be optimised in case T == vector<double>
      }
   }
   void Finalize()
   {
      fTo.Merge();
   }
   ~FillTOOperation()
   {
      Finalize();
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

   void Finalize()
   {
      auto rColl = fColls[0];
      for (unsigned int i = 1; i < fColls.size(); ++i) {
         auto& coll = fColls[i];
         for (T &v : *coll) {
            rColl->emplace_back(v);
         }
      }
   }
   ~TakeOperation()
   {
      Finalize();
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

   void Finalize()
   {
      ULong64_t totSize = 0;
      for (auto& coll : fColls) totSize += coll->size();
      auto rColl = fColls[0];
      rColl->reserve(totSize);
      for (unsigned int i = 1; i < fColls.size(); ++i) {
         auto& coll = fColls[i];
         rColl->insert(rColl->end(), coll->begin(), coll->end());
      }
   }

   ~TakeOperation()
   {
      Finalize();
   }
};

template<typename F, typename T>
class ReduceOperation {
   F fReduceFun;
   T* fReduceRes;
   std::vector<T> fReduceObjs;
public:
   ReduceOperation(F&& f, T* reduceRes, unsigned int nSlots) : fReduceFun(f),
      fReduceRes(reduceRes), fReduceObjs(nSlots, *reduceRes)
   { }

   void Exec(const T& value, unsigned int slot)
   {
      fReduceObjs[slot] = fReduceFun(fReduceObjs[slot], value);
   }

   void Finalize()
   {
      for (auto& t : fReduceObjs)
         *fReduceRes = fReduceFun(*fReduceRes, t);
   }

   ~ReduceOperation()
   {
      Finalize();
   }
};

class MinOperation {
   double *fResultMin;
   std::vector<double> fMins;

public:
   MinOperation(double *minVPtr, unsigned int nSlots);
   void Exec(double v, unsigned int slot);

   template <typename T, typename std::enable_if<TIsContainer<T>::fgValue, int>::type = 0>
   void Exec(const T &vs, unsigned int slot)
   {
      for (auto &&v : vs) fMins[slot] = std::min((double)v, fMins[slot]);
   }

   void Finalize();
   ~MinOperation();
};

extern template void MinOperation::Exec(const std::vector<float>&, unsigned int);
extern template void MinOperation::Exec(const std::vector<double>&, unsigned int);
extern template void MinOperation::Exec(const std::vector<char>&, unsigned int);
extern template void MinOperation::Exec(const std::vector<int>&, unsigned int);
extern template void MinOperation::Exec(const std::vector<unsigned int>&, unsigned int);

class MaxOperation {
   double *fResultMax;
   std::vector<double> fMaxs;

public:
   MaxOperation(double *maxVPtr, unsigned int nSlots);
   void Exec(double v, unsigned int slot);

   template <typename T, typename std::enable_if<TIsContainer<T>::fgValue, int>::type = 0>
   void Exec(const T &vs, unsigned int slot)
   {
      for (auto &&v : vs) fMaxs[slot] = std::max((double)v, fMaxs[slot]);
   }

   void Finalize();
   ~MaxOperation();
};

extern template void MaxOperation::Exec(const std::vector<float>&, unsigned int);
extern template void MaxOperation::Exec(const std::vector<double>&, unsigned int);
extern template void MaxOperation::Exec(const std::vector<char>&, unsigned int);
extern template void MaxOperation::Exec(const std::vector<int>&, unsigned int);
extern template void MaxOperation::Exec(const std::vector<unsigned int>&, unsigned int);


class MeanOperation {
   double *fResultMean;
   std::vector<Count_t> fCounts;
   std::vector<double> fSums;

public:
   MeanOperation(double *meanVPtr, unsigned int nSlots);
   void Exec(double v, unsigned int slot);

   template <typename T, typename std::enable_if<TIsContainer<T>::fgValue, int>::type = 0>
   void Exec(const T &vs, unsigned int slot)
   {
      for (auto &&v : vs) {
         fSums[slot] += v;
         fCounts[slot]++;
      }
   }

   void Finalize();
   ~MeanOperation();
};

extern template void MeanOperation::Exec(const std::vector<float>&, unsigned int);
extern template void MeanOperation::Exec(const std::vector<double>&, unsigned int);
extern template void MeanOperation::Exec(const std::vector<char>&, unsigned int);
extern template void MeanOperation::Exec(const std::vector<int>&, unsigned int);
extern template void MeanOperation::Exec(const std::vector<unsigned int>&, unsigned int);


} // end of NS Operations

} // end of NS Internal

} // end of NS ROOT

/// \endcond

#endif
