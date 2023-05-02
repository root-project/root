
/**
 \file ROOT/RDF/CUDAFillHelper.hxx
 \ingroup dataframe
 \author Jolly Chen, CERN
 \date 2023-04
*/

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_CUDAFILLHELPER
#define ROOT_CUDAFILLHELPER

#include <ROOT/RDF/RAction.hxx>
#include "ROOT/RDF/RActionImpl.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "RHnCUDA.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TStatistic.h"

#include <vector>
#include <string>
#include <array>

namespace ROOT {
namespace Internal {
namespace RDF {
using Hist_t = ::TH1D;

// clang-format off
static constexpr size_t getHistDim(TH3 *) { return 3; }
static constexpr size_t getHistDim(TH2 *) { return 2; }
static constexpr size_t getHistDim(TH1 *) { return 1; }

static constexpr char getHistType(TH1C *) { return (char) 0; }
static constexpr char getHistType(TH2C *) { return (char) 0; }
static constexpr char getHistType(TH3C *) { return (char) 0; }
static constexpr short getHistType(TH1S *) { return (short) 0; }
static constexpr short getHistType(TH2S *) { return (short) 0; }
static constexpr short getHistType(TH3S *) { return (short) 0; }
static constexpr int getHistType(TH1I *) { return 0; }
static constexpr int getHistType(TH2I *) { return 0; }
static constexpr int getHistType(TH3I *) { return 0; }
static constexpr float getHistType(TH1F *) { return (float) 0; }
static constexpr float getHistType(TH2F *) { return (float) 0; }
static constexpr float getHistType(TH3F *) { return (float) 0; }
static constexpr double getHistType(TH1D *) { return (double) 0; }
static constexpr double getHistType(TH2D *) { return (double) 0; }
static constexpr double getHistType(TH3D *) { return (double) 0; }

// clang-format on

template <typename HIST = Hist_t>
class R__CLING_PTRCHECK(off) CUDAFillHelper : public RActionImpl<CUDAFillHelper<HIST>> {
   static constexpr size_t dim = getHistDim((HIST *)nullptr);

   std::vector<HIST *> fObjects;
   std::vector<CUDAhist::RHnCUDA<decltype(getHistType((HIST *)nullptr)), dim> *> fCudaHist;

   template <typename H = HIST, typename = decltype(std::declval<H>().Reset())>
   void ResetIfPossible(H *h)
   {
      h->Reset();
   }

   void ResetIfPossible(TStatistic *h) { *h = TStatistic(); }

   // cannot safely re-initialize variations of the result, hence error out
   void ResetIfPossible(...)
   {
      throw std::runtime_error(
         "A systematic variation was requested for a custom Fill action, but the type of the object to be filled does"
         "not implement a Reset method, so we cannot safely re-initialize variations of the result. Aborting.");
   }

   void UnsetDirectoryIfPossible(TH1 *h) { h->SetDirectory(nullptr); }

   void UnsetDirectoryIfPossible(...) {}

   template <size_t DIMW>
   void FillWithWeight(unsigned int slot, const std::array<double, DIMW> &v)
   {
      double w = v.back();
      std::array<double, DIMW - 1> coords;
      std::copy(v.begin(), v.end() - 1, coords.begin());
      fCudaHist[slot]->Fill(coords, w);
   }

   template <typename... Coords>
   void FillWithoutWeight(unsigned int slot, const Coords &...x)
   {
      fCudaHist[slot]->Fill({x...});
   }

   // Merge overload for types with Merge(TCollection*), like TH1s
   template <typename H, typename = std::enable_if_t<std::is_base_of<TObject, H>::value, int>>
   auto Merge(std::vector<H *> &objs, int /*toincreaseoverloadpriority*/)
      -> decltype(objs[0]->Merge((TCollection *)nullptr), void())
   {
      TList l;
      for (auto it = ++objs.begin(); it != objs.end(); ++it)
         l.Add(*it);
      objs[0]->Merge(&l);
   }

   // Merge overload for types with Merge(const std::vector&)
   template <typename H>
   auto Merge(std::vector<H *> &objs, double /*toloweroverloadpriority*/)
      -> decltype(objs[0]->Merge(std::vector<HIST *>{}), void())
   {
      objs[0]->Merge({++objs.begin(), objs.end()});
   }

   // Merge overload to error out in case no valid HIST::Merge method was detected
   template <typename T>
   void Merge(T, ...)
   {
      static_assert(
         sizeof(T) < 0,
         "The type passed to Fill does not provide a Merge(TCollection*) or Merge(const std::vector&) method.");
   }

   // class which wraps a pointer and implements a no-op increment operator
   template <typename T>
   class ScalarConstIterator {
      const T *obj_;

   public:
      ScalarConstIterator(const T *obj) : obj_(obj) {}
      const T &operator*() const { return *obj_; }
      ScalarConstIterator<T> &operator++() { return *this; }
   };

   // helper functions which provide one implementation for scalar types and another for containers
   // TODO these could probably all be replaced by inlined lambdas and/or constexpr if statements
   // in c++17 or later

   // return unchanged value for scalar
   template <typename T, std::enable_if_t<!IsDataContainer<T>::value, int> = 0>
   ScalarConstIterator<T> MakeBegin(const T &val)
   {
      return ScalarConstIterator<T>(&val);
   }

   // return iterator to beginning of container
   template <typename T, std::enable_if_t<IsDataContainer<T>::value, int> = 0>
   auto MakeBegin(const T &val)
   {
      return std::begin(val);
   }

   // return 1 for scalars
   template <typename T, std::enable_if_t<!IsDataContainer<T>::value, int> = 0>
   std::size_t GetSize(const T &)
   {
      return 1;
   }

   // return container size
   template <typename T, std::enable_if_t<IsDataContainer<T>::value, int> = 0>
   std::size_t GetSize(const T &val)
   {
#if __cplusplus >= 201703L
      return std::size(val);
#else
      return val.size();
#endif
   }

   template <std::size_t ColIdx, typename End_t, typename... Its>
   void ExecLoop(unsigned int slot, End_t end, Its... its)
   {
      auto *thisSlotH = fObjects[slot];
      // loop increments all of the iterators while leaving scalars unmodified
      // TODO this could be simplified with fold expressions or std::apply in C++17
      auto nop = [](auto &&...) {};
      for (; GetNthElement<ColIdx>(its...) != end; nop(++its...)) {
         thisSlotH->Fill(*its...);
      }
   }

public:
   CUDAFillHelper(CUDAFillHelper &&) = default;
   CUDAFillHelper(const CUDAFillHelper &) = delete;

   // Initialize fCudaHist
   inline void init_cuda(HIST *obj, int i)
   {
      if (getenv("DBG"))
         printf("Init cuda hist %d\n", i);
      auto dims = obj->GetDimension();
      std::array<Int_t, dim> ncells;
      std::array<Double_t, dim> xlow;
      std::array<Double_t, dim> xhigh;
      std::array<const Double_t *, dim> binEdges;
      TAxis *ax;

      for (auto d = 0; d < dims; d++) {
         if (d == 0) {
            ax = obj->GetXaxis();
         } else if (d == 1) {
            ax = obj->GetYaxis();
         } else {
            ax = obj->GetZaxis();
         }

         ncells[d] = ax->GetNbins() + 2;
         binEdges[d] = ax->GetXbins()->GetArray();
         xlow[d] = ax->GetXmin();
         xhigh[d] = ax->GetXmax();

         if (getenv("DBG"))
            printf("\tdim %d --- nbins: %d xlow: %f xhigh: %f\n", d, ncells[d], xlow[d], xhigh[d]);
      }

      fCudaHist[i] =
         new CUDAhist::RHnCUDA<decltype(getHistType((HIST *)nullptr)), dim>(ncells, xlow, xhigh, binEdges.data());
   }

   CUDAFillHelper(const std::shared_ptr<HIST> &h, const unsigned int nSlots)
      : fObjects(nSlots, nullptr), fCudaHist(nSlots, nullptr)
   {
      fObjects[0] = h.get();
      init_cuda(fObjects[0], 0);

      // Initialize all other slots
      for (unsigned int i = 1; i < nSlots; ++i) {
         fObjects[i] = new HIST(*fObjects[0]);
         UnsetDirectoryIfPossible(fObjects[i]);
         init_cuda(fObjects[i], i);
      }
   }

   void InitTask(TTreeReader *, unsigned int) {}

   template <typename... ValTypes, std::enable_if_t<!Disjunction<IsDataContainer<ValTypes>...>::value, int> = 0>
   auto Exec(unsigned int slot, const ValTypes &...x) -> decltype(fObjects[slot]->Fill(x...), void())
   {
      if constexpr (sizeof...(ValTypes) > dim)
         FillWithWeight<dim + 1>(slot, {((Double_t)x)...});
      else
         FillWithoutWeight(slot, x...);
      return;

      fObjects[slot]->Fill(x...);
   }

   // at least one container argument
   template <typename... Xs, std::enable_if_t<Disjunction<IsDataContainer<Xs>...>::value, int> = 0>
   auto Exec(unsigned int slot, const Xs &...xs) -> decltype(fObjects[slot]->Fill(*MakeBegin(xs)...), void())
   {
      // array of bools keeping track of which inputs are containers
      constexpr std::array<bool, sizeof...(Xs)> isContainer{IsDataContainer<Xs>::value...};

      // index of the first container input
      constexpr std::size_t colidx = FindIdxTrue(isContainer);
      // if this happens, there is a bug in the implementation
      static_assert(colidx < sizeof...(Xs), "Error: index of collection-type argument not found.");

      // get the end iterator to the first container
      auto const xrefend = std::end(GetNthElement<colidx>(xs...));

      // array of container sizes (1 for scalars)
      std::array<std::size_t, sizeof...(xs)> sizes = {{GetSize(xs)...}};

      for (std::size_t i = 0; i < sizeof...(xs); ++i) {
         if (isContainer[i] && sizes[i] != sizes[colidx]) {
            throw std::runtime_error("Cannot fill histogram with values in containers of different sizes.");
         }
      }

      ExecLoop<colidx>(slot, xrefend, MakeBegin(xs)...);
   }

   template <typename T = HIST>
   void Exec(...)
   {
      static_assert(sizeof(T) < 0,
                    "When filling an object with RDataFrame (e.g. via a Fill action) the number or types of the "
                    "columns passed did not match the signature of the object's `Fill` method.");
   }

   void Initialize()
   { /* noop */
   }

   void Finalize()
   {
      Double_t stats[13];

      for (unsigned int i = 0; i < fObjects.size(); ++i) {
         HIST *h = fObjects[i];
         fCudaHist[i]->RetrieveResults(h->GetArray(), stats);
         h->SetStatsData(stats);
         h->SetEntries(fCudaHist[i]->GetEntries());
         // printf("%d %d??\n", fObjects[i]->GetArray()->size(), fObjects[i]->GetXaxis()->GetNbins());
         if (getenv("DBG")) {
            printf("cuda stats:");
            for (int j = 0; j < 13; j++) {
               printf("%f ", stats[j]);
            }
            printf(" %f\n", fObjects[0]->GetEntries());
         }
      }

      if (getenv("DBG")) {
         fObjects[0]->GetStats(stats);
         printf("stats:");
         for (int j = 0; j < 13; j++) {
            printf("%f ", stats[j]);
         }
         printf(" %f\n", fObjects[0]->GetEntries());
         if (getenv("DBG") && atoi(getenv("DBG")) > 1) {

            printf("histogram:");
            for (int j = 0; j < fObjects[0]->GetNcells(); ++j) {
               printf("%f ", fObjects[0]->GetArray()[j]);
            }
            printf("\n");
         }
      }

      if (fObjects.size() == 1)
         return;

      Merge(fObjects, /*toselectcorrectoverload=*/0);

      // delete the copies we created for the slots other than the first
      for (auto it = ++fObjects.begin(); it != fObjects.end(); ++it)
         delete *it;
   }

   HIST &PartialUpdate(unsigned int slot)
   {
      return *fObjects[slot];
   }

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      return std::make_unique<RMergeableFill<HIST>>(*fObjects[0]);
   }

   // if the fObjects vector type is derived from TObject, return the name of the object
   template <typename T = HIST, std::enable_if_t<std::is_base_of<TObject, T>::value, int> = 0>
   std::string GetActionName()
   {
      return std::string(fObjects[0]->IsA()->GetName()) + "\\n" + std::string(fObjects[0]->GetName());
   }

   // if fObjects is not derived from TObject, indicate it is some other object
   template <typename T = HIST, std::enable_if_t<!std::is_base_of<TObject, T>::value, int> = 0>
   std::string GetActionName()
   {
      return "Fill CUDA histogram";
   }

   template <typename H = HIST>
   CUDAFillHelper MakeNew(void *newResult)
   {
      auto &result = *static_cast<std::shared_ptr<H> *>(newResult);
      ResetIfPossible(result.get());
      UnsetDirectoryIfPossible(result.get());
      return CUDAFillHelper(result, fObjects.size());
   }
};

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif
